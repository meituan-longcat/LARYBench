import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def split_dataset(input_file, train_output, val_output, stratify):
    # 1. 读取数据
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return

    print(f"正在处理数据，总行数: {len(df)}")

    # 2. 划分数据集
    # test_size=0.25 对应 2.5/(7.5+2.5) 的比例
    # stratify=df['action'] 核心参数：保证训练集和验证集中 action 的类别分布一致
    if stratify:
        train_df, val_df = train_test_split(
            df, 
            test_size=0.3, 
            stratify=df['action'], 
            random_state=42, # 固定随机种子，保证每次运行结果一样
            shuffle=True
        )
    else:
        train_df, val_df = train_test_split(
            df, 
            test_size=0.3, 
            random_state=42, # 固定随机种子，保证每次运行结果一样
            shuffle=True
        )
    # 3. 保存结果
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)

    # 4. 打印验证信息
    print("-" * 30)
    print(f"划分完成！")
    print(f"训练集: {len(train_df)} 条 -> {train_output}")
    print(f"验证集: {len(val_df)} 条 -> {val_output}")
    
    print("-" * 30)
    print("分布一致性检查 (前5个类别):")
    print("\n[原始分布比例]:")
    print(df['action'].value_counts(normalize=True))
    print("\n[验证集分布比例]:")
    print(val_df['action'].value_counts(normalize=True))

if __name__ == "__main__":
    # 配置输入输出文件名
    # DATASET = 'human_1st'
    # MODEL = 'flux2'
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--stratify", action="store_true", help="是否分层")
    args = parser.parse_args()

    split_dataset(args.input_csv, args.input_csv.replace('all_', 'train_'), args.input_csv.replace('all_', 'val_'), args.stratify)