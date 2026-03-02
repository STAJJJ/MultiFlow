import pickle
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from uer.layers import *
from uer.encoders import *
from uer.utils.vocab import Vocab
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
import tqdm
import csv
import numpy as np
import os
import pdb
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class Classifier(nn.Module):
    def __init__(self, args, len_embedding_matrix, iat_embedding_matrix):
        """
        len_embedding_matrix: numpy array [len_vocab_size, hidden_size]
        iat_embedding_matrix: numpy array [iat_vocab_size, hidden_size]
        """
        super(Classifier, self).__init__()
        #pdb.set_trace()

        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)

        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha

        self.hidden_size = args.hidden_size
        packet_num = getattr(args, "packet_num", 8)

        
        len_weight = torch.tensor(len_embedding_matrix, dtype=torch.float32)
        iat_weight = torch.tensor(iat_embedding_matrix, dtype=torch.float32)

        self.length_emb = nn.Embedding.from_pretrained(len_weight, freeze=False)
        self.time_emb = nn.Embedding.from_pretrained(iat_weight, freeze=False)

        dir_matrix = np.array([
            np.random.normal(0, 0.02, size=(len_weight.shape[1],)),
            np.random.normal(0, 0.02, size=(len_weight.shape[1],)),
            [0]*len_weight.shape[1]
        ]
        )

        dir_weight = torch.tensor(dir_matrix, dtype=torch.float32)

        self.dir_emb = nn.Embedding.from_pretrained(dir_weight, freeze=False)
      

        self.payload_fc = nn.Linear(self.hidden_size, 512)

        self.stat_cnn = nn.Conv2d(3, 1, kernel_size=(packet_num, 1))  
        self.stat_fc = nn.Linear(300, 512)


        self.attention_fc = nn.Linear(512 * 2, 1)  


        self.classifier = nn.Sequential(
            nn.Linear(512, self.hidden_size),
            # nn.BatchNorm1d(self.hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size // 2, self.labels_num)
        )

    def forward(self, src, tgt, seg, soft_tgt=None, length_idx=None, time_idx=None, direction_idx=None):
        mode = getattr(self, "ablation_mode", "full")
        if mode in ["full", "payload"]:
            emb = self.embedding(src, seg)
            enc_out = self.encoder(emb, seg)
            if self.pooling == "mean":
                payload_vec = torch.mean(enc_out, dim=1)
            elif self.pooling == "max":
                payload_vec = torch.max(enc_out, dim=1)[0]
            elif self.pooling == "last":
                payload_vec = enc_out[:, -1, :]
            else:
                payload_vec = enc_out[:, 0, :]

            payload_vec = torch.tanh(self.payload_fc(payload_vec))  # [B, 512]
        else:
            payload_vec = torch.zeros((src.size(0), 512), device=src.device)

        if mode in ["full", "stat"] and (length_idx is not None and time_idx is not None and direction_idx is not None):
            len_emb = self.length_emb(length_idx)    
            time_emb = self.time_emb(time_idx)   
            dir_emb = self.dir_emb(direction_idx) 
            stat_input = torch.stack([len_emb, time_emb, dir_emb], dim=1)
            stat_out = self.stat_cnn(stat_input).squeeze(2).squeeze(1)
            stat_vec = torch.tanh(self.stat_fc(stat_out))  # [B, 512]
        else:
            stat_vec = torch.zeros_like(payload_vec)
        if mode == "payload":
            fusion_vec = payload_vec
        elif mode == "stat":
            fusion_vec = stat_vec
        else: 
            fusion_input = torch.cat([payload_vec, stat_vec], dim=1)
            alpha = torch.sigmoid(self.attention_fc(fusion_input))
            fusion_vec = alpha * payload_vec + (1 - alpha) * stat_vec
        logits = self.classifier(fusion_vec)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                    (1 - self.soft_alpha) * nn.NLLLoss()(F.log_softmax(logits, dim=-1), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(F.log_softmax(logits, dim=-1), tgt.view(-1))
            return loss, logits
        else:
            return None, logits
def count_labels_num(path):
    labels_set, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row_idx, row in enumerate(csv_reader, start=1):
            label = row[0].strip()
            labels_set.append(label)
    labels_set = list(set(labels_set))
    label_dict = {}
    for i in range(len(labels_set)):
        label_dict[labels_set[i]]=i
    return len(labels_set),label_dict
def load_or_initialize_parameters_with_path(model, model_path=None):
    if model_path is not None:
        pretrain_dict = torch.load(model_path, map_location='cuda:0')
        model.load_state_dict(torch.load(model_path, map_location='cuda:0'), strict=False)
    else:
        for n, p in model.named_parameters():
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)
                
def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)                             
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler

def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size: (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size:]
        seg_batch = seg[instances_num // batch_size * batch_size:, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size:, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
            
def read_dataset(args, path,length_dictionary,iat_dictionary,label_dict):
    with open(path, mode="r", encoding="utf-8") as f:
        csv_reader = csv.reader(f)   
        columns = next(csv_reader, []) 
        for row_idx, row in enumerate(csv_reader, start=1):
            tgt = row[0].strip()
            tgt =label_dict[tgt] 
            #pdb.set_trace()
            payloads=json.loads(row[-1])
            sos_token_id= args.tokenizer.convert_tokens_to_ids([CLS_TOKEN])
            segment_token_id=args.tokenizer.convert_tokens_to_ids(['[SEP]'])
            pad_token_id=args.tokenizer.convert_tokens_to_ids(['[PAD]'])    
            src_token_id=sos_token_id
            for packet_payload in payloads:
                payload=packet_payload.strip()
                packet_token=args.tokenizer.tokenize(packet_payload)
                packet_token_id=args.tokenizer.convert_tokens_to_ids(packet_token)
                src_token_id.extend(packet_token_id)
            seg = [1] * len(src_token_id)
            if len(src_token_id) >= args.seq_length:
                src_token_id = src_token_id[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src_token_id) < args.seq_length:
                src_token_id.append(pad_token_id[0])
                seg.append(0)        
            packet_num = getattr(args, "packet_num", 8)
            lengths = [str(i) for i in json.loads(row[1])]
            lengthids = [length_dictionary[i] if i in length_dictionary.keys() else length_dictionary['UNK'] for i in lengths ]
            if len(lengthids) >= packet_num:
                lengthids = lengthids[:packet_num]
            else:
                lengthids+=[length_dictionary['PAD']] * (packet_num - len(lengthids))  
            directions_dictionary={
                '1':0,
                "-1":1,
                "PAD":2
            }
            directions = [directions_dictionary[str(i)] for i in json.loads(row[2])]
            if len(directions) >= packet_num:
                directions = directions[:packet_num]
            else:
                directions += [2] * (packet_num - len(directions))
            iats = [str(i) for i in json.loads(row[3])]
            iats_ids = [iat_dictionary[i] if i in iat_dictionary.keys() else iat_dictionary['UNK'] for i in iats]
            if len(iats_ids) >= packet_num:
                iats_ids = iats_ids[:packet_num]
            else:
                iats_ids+=[iat_dictionary['PAD']] * (packet_num - len(iats_ids))
            entry = (src_token_id, tgt, seg, lengthids, iats_ids, directions)
            #import pdb
            #pdb.set_trace()
            dataset.append(entry)
    return dataset
class DataLoader(Dataset):
    def __init__(self, data,length_dictionary_path,iat_dictionary_path,packet_number=40):
        import pdb
        pdb.set_trace()
        self.data = data
        
        self.packet_number= packet_num
        try:
            self.length_dictionary=pickle.load(length_dictionary_path)
        except:
            self.length_dictionary={}
        try:
            self.iat_dictionary=pickle.load(iat_dictionary_path)
        except:
            self.iat_dictionary={}
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
def build_stat_indices(dataset, len_dict, iat_dict, len_emb_np, iat_emb_np, packet_num=40):
    for ex in dataset:
        l_seq = ex[3][:packet_num] + [0] * (packet_num - len(ex[3]))
        t_seq = ex[4][:packet_num] + [0] * (packet_num - len(ex[4]))
        d_seq = ex[5][:packet_num] + [0] * (packet_num - len(ex[5]))

        l_idx_seq = []
        for l in l_seq:
            if l not in len_dict:
                new_idx = len(len_dict)
                len_dict[l] = new_idx
                # append random vector
                new_emb = np.random.normal(0, 0.02, size=(len_emb_np.shape[1],))
                len_emb_np = np.vstack([len_emb_np, new_emb])
            l_idx_seq.append(len_dict[l])
        length_idx_list.append(l_idx_seq)
        t_idx_seq = []
        for t in t_seq:
            if t not in iat_dict:
                new_idx = len(iat_dict)
                iat_dict[t] = new_idx
                new_emb = np.random.normal(0, 0.02, size=(iat_emb_np.shape[1],))
                iat_emb_np = np.vstack([iat_emb_np, new_emb])
            t_idx_seq.append(iat_dict[t])
        time_idx_list.append(t_idx_seq)

        # map direction: -1 -> 0, 0 -> 1, 1 -> 2
        d_idx_seq = [0 if d == -1 else 2 if d == 1 else 1 for d in d_seq]
        direction_idx_list.append(d_idx_seq)

    length_idx = torch.LongTensor(length_idx_list)   # [N, packet_num]
    time_idx = torch.LongTensor(time_idx_list)       # [N, packet_num]
    direction_idx = torch.LongTensor(direction_idx_list)  # [N, packet_num]
    return length_idx, time_idx, direction_idx, len_emb_np, iat_emb_np
    
def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch,
                length_idx_batch=None, time_idx_batch=None, dir_idx_batch=None, soft_tgt_batch=None):
    model.zero_grad()
    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if length_idx_batch is not None:
        length_idx_batch = length_idx_batch.to(args.device)
        time_idx_batch = time_idx_batch.to(args.device)
        dir_idx_batch = dir_idx_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch,
                    length_idx=length_idx_batch, time_idx=time_idx_batch, direction_idx=dir_idx_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)
    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()
    optimizer.step()
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    return loss,current_lr

def evaluate(args, dataset, print_confusion_matrix=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])
    length_idx = torch.LongTensor([sample[3] for sample in dataset])
    time_idx = torch.LongTensor([sample[4] for sample in dataset])
    direction_idx = torch.LongTensor([sample[5] for sample in dataset])
    batch_size = args.batch_size
    correct = 0
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    args.model.eval()
    total_batches = (src.size(0) + batch_size - 1) // batch_size
    for i in range(total_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, src.size(0))
        src_batch = src[start:end].to(args.device)
        tgt_batch = tgt[start:end].to(args.device)
        seg_batch = seg[start:end].to(args.device)
        len_batch = length_idx[start:end].to(args.device)
        time_batch = time_idx[start:end].to(args.device)
        dir_batch = direction_idx[start:end].to(args.device)
        with torch.no_grad():
            _, logits = args.model(
                src_batch, tgt_batch, seg_batch,
                length_idx=len_batch,
                time_idx=time_batch,
                direction_idx=dir_batch
            )
        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
        gold = tgt_batch
        for j in range(pred.size(0)):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()
    metrics_dict = {}
    eps = 1e-9
    for i in range(args.labels_num):
        p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
        metrics_dict[f"Label_{i}"] = {
            "precision": p,
            "recall": r,
            "f1": f1
        }

    macro_recall = sum([v['recall'] for v in metrics_dict.values()]) / args.labels_num
    macro_f1 = sum([v['f1'] for v in metrics_dict.values()]) / args.labels_num
    accuracy = correct / len(dataset)

    if print_confusion_matrix:
        out_path = "/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/datasets/own_lyj/CSTNET/re_2/all/confusion_matrix_2.14_512_2e-5_25.tsv"

        with open(out_path, 'w') as f:
            f.write("===== Experiment Configuration =====\n")
            f.write(f"learning_rate\t{args.learning_rate}\n")
            f.write(f"batch_size\t{args.batch_size}\n")
            f.write(f"seq_length\t{args.seq_length}\n")
            f.write(f"epochs\t{args.epochs_num}\n")
            f.write(f"hidden_size\t{args.hidden_size}\n")
            f.write(f"packet_num\t{getattr(args,'packet_num',40)}\n")
            f.write(f"labels_num\t{args.labels_num}\n")
            f.write(f"ablation_mode\t{args.ablation_mode}\n")
            f.write(f"pooling\t{args.pooling}\n")
            f.write(f"optimizer\t{args.optimizer}\n")
            f.write(f"scheduler\t{args.scheduler}\n")
            f.write(f"device\t{args.device}\n")
            f.write(f"model_path\t{args.output_model_path}\n")
            f.write("\n")
            f.write("===== Confusion Matrix (rows=predicted, cols=true) =====\n")
            f.write("\t" + "\t".join([f"Label_{i}" for i in range(args.labels_num)]) + "\n")

            for i in range(args.labels_num):
                f.write(
                    f"Label_{i}\t" +
                    "\t".join(str(x.item()) for x in confusion[i]) +
                    "\n"
                )
            f.write("\n===== Precision / Recall / F1 =====\n")
            for label, scores in metrics_dict.items():
                f.write(
                    f"{label}\t"
                    f"P={scores['precision']:.6f}\t"
                    f"R={scores['recall']:.6f}\t"
                    f"F1={scores['f1']:.6f}\n"
                )
            f.write("\n===== Overall =====\n")
            f.write(f"Accuracy\t{accuracy:.6f}\n")
            f.write(f"Macro Recall\t{macro_recall:.6f}\n")
            f.write(f"Macro F1\t{macro_f1:.6f}\n")

    print(f"Overall Accuracy: {accuracy:.4f}, Macro Recall={macro_recall:.4f}, Macro F1={macro_f1:.4f}")
    return accuracy, confusion, metrics_dict,macro_f1

    
def main():
    import matplotlib.pyplot as plt
    import io
    import torchvision
    import PIL.Image
    import pdb
    from torch.utils.tensorboard import SummaryWriter
    log_dir = "/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/datasets/own_lyj/CSTNET/log/CSTTLSAll/log_2_13"
    writer = SummaryWriter(log_dir)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)
    print("开始加载数据集")

    parser.set_defaults(
        train_path="/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/datasets/own_lyj/CSTNET/data_2_10/splits/train.csv",
        dev_path="/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/datasets/own_lyj/CSTNET/data_2_10/splits/valid.csv",
        test_path="/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/datasets/own_lyj/CSTNET/data_2_10/splits/test.csv",
        vocab_path="/3241903007/workstation/AnomalyTrafficDetection/ET-BERT/models/encryptd_vocab.txt",
        length_emb_path="/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/wordembedding/len_embedding.npy",
        time_emb_path="/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/wordembedding/iat_embedding.npy",
        len_dict_path="/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/wordembedding/len_dict.pkl",
        iat_dict_path="/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/wordembedding/iat_dict.pkl",
    )

    parser.add_argument("--ablation_mode", choices=["full", "payload", "stat"], default="full")
    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first")
    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert")
    parser.add_argument("--soft_targets", action='store_true')
    parser.add_argument("--soft_alpha", type=float, default=0.5)
    parser.add_argument("--reg_lambda", type=float, default=0.0, help="Alpha regularization strength.")
    args = parser.parse_args()
    print(f"初始学习率 (--learning_rate): {args.learning_rate:.6f}")
    
    args.output_model_path = "/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/datasets/own_lyj/CSTNET/model/CENTLSAll.bin"
    model_path = "/3241903007/workstation/AnomalyTrafficDetection/ET-BERT/models/pre-trained_model.bin"
    BEST_MODEL_TEMP_PATH = args.output_model_path + ".best_temp"
    args = load_hyperparam(args)
    set_seed(args.seed)
    args.labels_num,label_dict = count_labels_num(args.train_path)
    print(label_dict)
    # import pdb;pdb.set_trace()
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    with open(args.len_dict_path, "rb") as f:
        length_vocab = pickle.load(f)
    with open(args.iat_dict_path, "rb") as f:
        time_vocab = pickle.load(f)

    len_embedding = np.load(args.length_emb_path)
    iat_embedding = np.load(args.time_emb_path)
    trainset = read_dataset(args, args.train_path,length_vocab,time_vocab,label_dict)
    devset = read_dataset(args, args.dev_path,length_vocab,time_vocab,label_dict)
    testset = read_dataset(args, args.test_path,length_vocab,time_vocab,label_dict) if args.test_path else None
    print(f"Train: {len(trainset)} | Dev: {len(devset)} | Test: {len(testset) if testset else 0}")
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size
    print("=" * 50)
    print(f"初始学习率 (--learning_rate): {args.learning_rate:.6f}")
    print(f"Training batch_size: {batch_size}")
    print(f"Training seq_length: {args.seq_length}")
    print("=" * 50)


    
    model = Classifier(args, len_embedding, iat_embedding)
    #print(model)
    model.ablation_mode = args.ablation_mode
    #pdb.set_trace()
    load_or_initialize_parameters_with_path(model, model_path=model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1
    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    args.model = model
    best_dev_F1 = 0.0
    best_epoch = 0
    
    print("Start training.")
    
    src = torch.LongTensor([ex[0] for ex in trainset])
    tgt = torch.LongTensor([ex[1] for ex in trainset])
    seg = torch.LongTensor([ex[2] for ex in trainset])
    length_idx = torch.LongTensor([ex[3] for ex in trainset])
    time_idx = torch.LongTensor([ex[4] for ex in trainset])
    direction_idx = torch.LongTensor([ex[5] for ex in trainset])
    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()
        total_loss = 0.0
        num_batches = (src.size(0) + batch_size - 1) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, src.size(0))
            src_batch = src[start:end]
            tgt_batch = tgt[start:end]
            seg_batch = seg[start:end]
            len_batch = length_idx[start:end]
            time_batch = time_idx[start:end]
            dir_batch = direction_idx[start:end]

            loss, current_lr = train_model(args, model, optimizer, scheduler,
                               src_batch, tgt_batch, seg_batch,
                               length_idx_batch=len_batch, time_idx_batch=time_batch, dir_idx_batch=dir_batch)
            if i%10==0:
                print(f"epoches:{epoch},steps:{i},loss:{loss.item()},lr:{current_lr}")
            total_loss += loss.item()

            writer.add_scalar("Loss/train", loss.item(), epoch * num_batches + i)
            writer.add_scalar("LearningRate/train", current_lr, epoch * num_batches + i)
        acc, confusion, metrics_dict, macro_f1 = evaluate(args, devset)
        
        print(f"Epoch {epoch} Dev Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")
        dev_F1 = macro_f1
        writer.add_scalar("F1/dev", dev_F1, epoch)
        for i in range(args.labels_num):
            eps = 1e-9
            p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
            r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
            f1 = 0 if (p + r) == 0 else 2 * p * r / (p + r)
            writer.add_scalar(f"Precision/Label_{i}", p, epoch)
            writer.add_scalar(f"Recall/Label_{i}", r, epoch)
            writer.add_scalar(f"F1/Label_{i}", f1, epoch)
        if dev_F1 > best_dev_F1:
            best_dev_F1 = dev_F1
            best_epoch = epoch
            print(f"--- Epoch {epoch}: New best Dev F1: {best_dev_F1:.4f}. Saving best model temporarily to {BEST_MODEL_TEMP_PATH} ---")
            
            if torch.cuda.device_count() > 1:
                save_model(model.module, BEST_MODEL_TEMP_PATH)
            else:
                save_model(model, BEST_MODEL_TEMP_PATH)
    if testset is not None:
        print("\nTest set evaluation.")
        print(f"Loading best model from Epoch {best_epoch} with Dev F1: {best_dev_F1:.4f}")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(BEST_MODEL_TEMP_PATH))
            test_model = model.module
        else:
            model.load_state_dict(torch.load(BEST_MODEL_TEMP_PATH))
            test_model = model
            
        acc, confusion, metrics_dict, macro_f1 = evaluate(args, testset, print_confusion_matrix=True)
        writer.add_scalar("Accuracy/test", acc, 0)
        writer.add_scalar("F1/test", macro_f1, 0)
        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.matshow(confusion.numpy(), cmap="Blues")
        plt.colorbar(cax)
        ax.set_xlabel("True label")
        ax.set_ylabel("Predicted label")
        ax.set_title("Confusion matrix (Test)")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = torchvision.transforms.ToTensor()(image)
        writer.add_image("ConfusionMatrix/test", image, 0)
        buf.close()

    writer.close()

if __name__ == "__main__":
    main()
