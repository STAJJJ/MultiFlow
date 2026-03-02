#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from flowcontainer.extractor import extract
import scapy.all as scapy
import math
import pdb
from multiprocessing import Pool
import csv
import json
import binascii
import random
random.seed(40)
# ========= 参数配置 =========
pcap_root = "/3241903007/workstation/AnomalyTrafficDetection/dataset/ISCX-VPN-NonVPN-2016/datasets/flows"

input_dirs = [os.path.join(pcap_root, d) for d in os.listdir(pcap_root) 
              if os.path.isdir(os.path.join(pcap_root, d))]

output_dir = "/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/datasets/own_lyj/ISCX-VPN-app/data_2_13"

output_csv = os.path.join(output_dir, "all_flows.csv")
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

if not os.path.exists(output_csv):
    with open(output_csv, "w", newline="") as f:
        csv.writer(f).writerow(["label","lengths","directions","iats","payloads"])




max_packets = 40   
NUM_WORKERS = 64  

payload_len = 64 
payload_pac = 10
MAX_FLOWS_PER_LABEL = 500

# ========= 提取流特征 =========
def get_flow_features(pcap_path, max_packets=40):
    length_list=[]
    iat_list=[]
    dir_list=[]
    hexdata_list=[]
    total_packets=0
    print("pcap_file_path:-------------------------------------")
    print(pcap_path)
    try:
        packets = scapy.rdpcap(pcap_path)
        packets_info={}
        for packet in packets:
            packet_timestamp = f"{packet.time:.6f}"
            packet_hexdata=bytes(packet)
            hex_str = binascii.hexlify(packet_hexdata).decode("utf-8")
            packets_info[packet_timestamp]=hex_str
        print(f"packets length:{len(packets)}")
        if len(packets) < 0:
            return -1, 0, 0  # 返回包数为0

        try:
            feature_result = extract(
                pcap_path,
                filter='tcp',
                extension=["data.data"],
                split_flag=False
            )
            print("Extraction succeeded, flows:", len(feature_result))
        except Exception as e:
            print(f"Extraction failed for {pcap_path}: {e}")
            import pdb; pdb.set_trace()
        print("feature_result keys:", feature_result.keys())
        # pdb.set_trace()
        if not feature_result:
            feature_result = extract(pcap_path, filter='udp')
            if not feature_result:
                # pdb.set_trace()
                return -1, 0, 0
            
        for flow_key,flow_obj in feature_result.items():
            
            total_packets += len(flow_obj.ip_lengths)  # 实际包数
            lengths = np.array(flow_obj.ip_lengths[:max_packets]).tolist()
            lengths = [abs(i) for i in lengths]
            directions = [int(np.sign(x)) for x in flow_obj.ip_lengths[:max_packets]]
            timestamps = np.array(flow_obj.ip_timestamps[:max_packets])
            hexdata = [packets_info[f"{ts:.6f}"] for ts in timestamps]
            flow_data_list = []
            for i in range(min(payload_pac, len(hexdata))):
                packet_string = hexdata[i]
                packet_bigram = bigram_generation(packet_string, packet_len=payload_len, flag=True)
                flow_data_list.append(packet_bigram)                

            if len(timestamps)!=len(hexdata):
                pdb.set_trace()
            if len(lengths)<0:
                # pdb.set_trace()
                print(f"********************************************************************************")
                print(f"flow contains {len(lengths)} pcpas, less then min value 3, which will be filter")
                continue

            iats = np.diff(timestamps).tolist() if len(timestamps) > 1 else [0]
            iats = [round(i*100,1) for i in iats]
            iats.insert(0,0.0)
            length_list.append(lengths)
            iat_list.append(iats)
            dir_list.append(directions)
            hexdata_list.append(flow_data_list)

            print(f"\n[DEBUG] flow_key: {flow_key}")
            print(f"Lengths: {lengths}")
            print(f"Directions: {directions}")
            print(f"IATs: {iats if 'iats' in locals() else 'not yet computed'}")
            print(f"Payload count: {len(flow_data_list)}")
            for idx in range(len(flow_data_list)):
                print(f"Payload {idx+1} (bigram): {flow_data_list[idx]}")
            # pdb.set_trace()
        return (length_list, dir_list, iat_list, hexdata_list), total_packets, pcap_path
    except Exception as e:
        # pdb.set_trace()
        return -1, 0, 0

def mark_processed(log_file_path, pcap_file):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, 'a') as f:
        json.dump({"file": pcap_file}, f)
        f.write("\n") 

    
def get_processed_pcapfile(log_file_path):
    processed_files = []
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    processed_files.append(obj["file"])
                except json.JSONDecodeError:
                    continue
    return processed_files


def cut(obj, sec):
    result = [obj[i:i+sec] for i in range(0,len(obj),sec)]
    try:
        remanent_count = len(result[0])%4
    except Exception as e:
        remanent_count = 0
        print("cut datagram error!")
    if remanent_count == 0:
        pass
    else:
        result = [obj[i:i+sec+remanent_count] for i in range(0,len(obj),sec+remanent_count)]
    return result

def bigram_generation(packet_datagram, packet_len=64, flag=True):
    result = ''
    generated_datagram = cut(packet_datagram,1)
    token_count = 0
    for sub_string_index in range(len(generated_datagram)):
        if sub_string_index != (len(generated_datagram) - 1):
            token_count += 1
            if token_count > packet_len:
                break
            else:
                merge_word_bigram = generated_datagram[sub_string_index] + generated_datagram[sub_string_index + 1]
        else:
            break
        result += merge_word_bigram
        result += ' '
    
    return result

def append_flow_csv(csv_path, label, lengths, directions, iats, payloads):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            label,
            json.dumps(lengths),
            json.dumps(directions),
            json.dumps(iats),
            json.dumps(payloads)
        ])




def main():
    processed_log = os.path.join(output_dir, "processed_files.json")
    processed_pcap_files = get_processed_pcapfile(processed_log)

    label_info = {}
    total_files = 0
    for input_dir in input_dirs:
        import pdb
        pdb.set_trace()
        label = os.path.basename(input_dir)
        label_info[label] = 0 

    for input_dir in input_dirs:
        label = os.path.basename(input_dir) 
        print(f"\n[+] Processing label: {label}")


        label_flow_cnt = 0


        all_pcaps = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".pcap"):
                    all_pcaps.append(os.path.join(root, file))

        # import pdb
        # pdb.set_trace()
        if len(all_pcaps) > MAX_FLOWS_PER_LABEL:
            selected_pcaps = random.sample(all_pcaps, MAX_FLOWS_PER_LABEL)
        else:
            selected_pcaps = all_pcaps

        for pcap_path in selected_pcaps:
            if pcap_path in processed_pcap_files:
                print(f"Already processed: {pcap_path}, skipping.")
                continue 

            print(f"Processing file: {pcap_path}")

            feature_results, total_pcaps, fpath = get_flow_features(pcap_path)
            print(f"Feature extraction result for {pcap_path}: {'Success' if feature_results != -1 else 'Failed'}")
            print(feature_results)

            if feature_results == -1:
                continue

            label_info[label] += 1
            total_files += 1

            lengths_list, dirs_list, iats_list, payloads_list = feature_results

            for idx in range(len(lengths_list)):
                append_flow_csv(
                    output_csv,
                    label=label,
                    lengths=lengths_list[idx],
                    directions=dirs_list[idx],
                    iats=iats_list[idx],
                    payloads=payloads_list[idx]
                )

                label_flow_cnt += 1

            mark_processed(processed_log, pcap_path)
            processed_pcap_files.append(pcap_path)

        print(f"[✓] Label {label} finished with {label_flow_cnt} flows")

    label_info_path = os.path.join(output_dir, "label_info.json")
    with open(label_info_path, "w") as f:
        json.dump({
            "total_files": total_files,
            "labels": label_info
        }, f, indent=4)

    print(f"[+] Label info saved to {label_info_path}")
    print(f"Total successful files: {total_files}")
    for label, count in label_info.items():
        print(f"{label}: {count} files")


if __name__ == "__main__":
    main()
