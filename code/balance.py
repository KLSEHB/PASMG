import pandas as pd
import random


def balance_dataset_inplace(file_path):
    # 读取CSV文件
    df = pd.read_csv(file_path)

    # 提取 processed_func 和 target 列
    funcs = df["processed_func"].tolist()
    labels = df["target"].tolist()

    # 统计正负样本数量
    positive_samples = [i for i, label in enumerate(labels) if label == 1]
    negative_samples = [i for i, label in enumerate(labels) if label == 0]

    num_positive = len(positive_samples)
    num_negative = len(negative_samples)

    print(f"原始数据集中正样本数量: {num_positive}")
    print(f"原始数据集中负样本数量: {num_negative}")

    # 根据正负样本数量进行调整
    if num_positive > num_negative:
        # 如果正样本多于负样本，随机删除一部分正样本
        to_remove = random.sample(positive_samples, num_positive - num_negative)
        print(f"正样本多于负样本，随机删除 {len(to_remove)} 个正样本。")
    elif num_negative > num_positive:
        # 如果负样本多于正样本，随机删除一部分负样本
        to_remove = random.sample(negative_samples, num_negative - num_positive)
        print(f"负样本多于正样本，随机删除 {len(to_remove)} 个负样本。")
    else:
        # 如果正负样本数量相等，则无需调整
        print("正负样本数量相等，无需调整。")
        return

    # 删除需要移除的样本索引
    df_balanced = df.drop(to_remove).reset_index(drop=True)

    # 确认调整后的正负样本数量
    balanced_labels = df_balanced["target"].tolist()
    balanced_positive = sum(1 for label in balanced_labels if label == 1)
    balanced_negative = sum(1 for label in balanced_labels if label == 0)

    print(f"调整后数据集中正样本数量: {balanced_positive}")
    print(f"调整后数据集中负样本数量: {balanced_negative}")

    # 将平衡后的数据集直接写回原始文件
    df_balanced.to_csv(file_path, index=False)
    print(f"已将平衡后的数据集覆盖写入到原始文件：{file_path}")


# 示例用法
file_path = "/home/liao/handledataset/different_models_dataset/LineVul/Devign/TestSetMoreThan_512.csv"  # 替换为你的CSV文件路径
balance_dataset_inplace(file_path)