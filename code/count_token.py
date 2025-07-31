import json
import argparse
from transformers import RobertaTokenizer


def analyze_all_files(files, tokenizer):
    total_tokens = 0
    over_510 = 0
    total_paths = 0

    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                for path_key in ['path1', 'path2', 'path3', 'path4']:
                    code = data.get(path_key, '')
                    tokens = tokenizer.tokenize(code)
                    token_count = len(tokens)

                    total_tokens += token_count
                    total_paths += 1
                    if token_count > 510:
                        over_510 += 1

    # 计算统计指标
    avg_tokens = total_tokens / total_paths if total_paths else 0
    under_510 = total_paths - over_510
    over_pct = (over_510 / total_paths) * 100 if total_paths else 0
    under_pct = (under_510 / total_paths) * 100 if total_paths else 0

    return {
        "total_samples": total_paths,
        "avg_tokens": avg_tokens,
        "over_510": (over_510, over_pct),
        "under_510": (under_510, under_pct)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", default="../pretrain-model/pdbert", help="分词器名称或路径")
    parser.add_argument("--train_file", default="../Dataset/Func_preprocessed/train.jsonl", help="训练集路径")
    parser.add_argument("--valid_file", default="../Dataset/Func_preprocessed/valid.jsonl", help="验证集路径")
    parser.add_argument("--test_file", default="../Dataset/Func_preprocessed/test.jsonl", help="测试集路径")
    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    files = [args.train_file, args.valid_file, args.test_file]

    stats = analyze_all_files(files, tokenizer)

    print("\n====== 合并统计结果 ======")
    print(f"总路径数: {stats['total_samples']}")
    print(f"平均token数: {stats['avg_tokens']:.2f}")
    print(f"超过510token的路径: {stats['over_510'][0]} ({stats['over_510'][1]:.2f}%)")
    print(f"≤510token的路径: {stats['under_510'][0]} ({stats['under_510'][1]:.2f}%)")


if __name__ == "__main__":
    main()