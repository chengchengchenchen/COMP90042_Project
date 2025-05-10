#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from typing import List, Tuple, Optional

def compute_metrics(
    gt: List[str],
    retrieved: List[str]
) -> Tuple[int, Optional[float], Optional[float]]:
    """
    计算命中数、Recall 和 Precision。
    
    Args:
        gt: ground-truth evidence ID 列表
        retrieved: top-100 检索到的 evidence ID 列表

    Returns:
        tp: 命中数（检索结果中也在 gt 中的数量）
        recall: tp / len(gt) （如果 gt 为空则返回 None）
        precision: tp / len(retrieved) （如果 retrieved 为空则返回 None）
    """
    set_gt = set(gt)
    set_ret = set(retrieved)
    tp = len(set_gt & set_ret)
    recall = tp / len(gt) if gt else None
    precision = tp / len(retrieved) if retrieved else None
    return tp, recall, precision

def main(train_claims_path: str, top100_path: str):
    # 1. 读入 JSON
    with open(train_claims_path, 'r', encoding='utf-8') as f:
        train_claims = json.load(f)
    with open(top100_path, 'r', encoding='utf-8') as f:
        top100 = json.load(f)

    recalls = []
    precisions = []

    # 2. 针对每个 claim 计算指标
    for claim_id, claim_info in train_claims.items():
        gt_list = claim_info.get("evidences", [])
        retrieved_list = top100.get(claim_id, {}).get("evidences", [])

        tp, recall, precision = compute_metrics(gt_list, retrieved_list)
        recalls.append(recall if recall is not None else 0.0)
        precisions.append(precision if precision is not None else 0.0)

        # print(f"{claim_id}: hit {tp}/{len(gt_list)} → Recall@100={recall:.3f} Precision@100={precision:.3f}")

    # 3. 计算并输出平均值
    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    print("\n=== Overall ===")
    print(f"Average Recall@100   : {avg_recall:.3f}")
    print(f"Average Precision@100: {avg_precision:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="计算 BM25 top-100 检索结果相对于 ground-truth 的覆盖率（Recall@100）与准确率（Precision@100）"
    )
    parser.add_argument(
        "--train_claims",
        type=str,
        default="./data/train-claims.json",
        help="包含 ground-truth evidences 的 JSON 文件路径"
    )
    parser.add_argument(
        "--top100",
        type=str,
        default="./data/train-claims-top100.json",
        help="BM25 检索 top-100 结果的 JSON 文件路径"
    )
    args = parser.parse_args()
    main(args.train_claims, args.top100)