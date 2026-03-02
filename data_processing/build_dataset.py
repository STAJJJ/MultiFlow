from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np
import os

# 参数配置
input_csv = "/3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/datasets/own_lyj/ISCX-Tor-new/data_2.8/all_flows.csv"
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1
random_state1 = 41
random_state2 = 42

# 输出目录
output_dir = os.path.join(os.path.dirname(input_csv), "splits")
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_csv)
X = df.drop(columns=["label"])
y = df["label"]

# 第一次划分 train vs temp
split_1 = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio+test_ratio, random_state=random_state1)
train_idx, temp_idx = next(split_1.split(X, y))
train_df = df.iloc[train_idx].reset_index(drop=True)
temp_df = df.iloc[temp_idx].reset_index(drop=True)

# ------------------ 处理 temp_df ------------------
# 将样本少于2的类别直接放入 test
value_counts = temp_df['label'].value_counts()
rare_labels = value_counts[value_counts < 2].index.tolist()

rare_df = temp_df[temp_df['label'].isin(rare_labels)].reset_index(drop=True)
remain_df = temp_df[~temp_df['label'].isin(rare_labels)].reset_index(drop=True)

if len(remain_df) == 0:
    # 没有剩余可划分的样本，全部放到 test
    valid_df = pd.DataFrame(columns=df.columns)
    test_df = temp_df
else:
    # 第二次划分 valid vs test
    split_2 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio/(valid_ratio+test_ratio), random_state=random_state2)
    valid_idx, test_idx = next(split_2.split(remain_df, remain_df['label']))

    valid_df = remain_df.iloc[valid_idx].reset_index(drop=True)
    test_df = remain_df.iloc[test_idx].reset_index(drop=True)

    # 加回少样本类别
    test_df = pd.concat([test_df, rare_df], ignore_index=True)

# 保存
train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
valid_df.to_csv(os.path.join(output_dir, "valid.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

print("划分完成:")
print(f"训练集: {len(train_df)} 流")
print(f"验证集: {len(valid_df)} 流")
print(f"测试集: {len(test_df)} 流")
print(f"CSV 文件已保存到: {output_dir}")
