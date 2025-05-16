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
    Args:
        gt: ground-truth evidence ID
        retrieved: top-100 evidence ID 

    Returns:
        tp: hit
        recall: tp / len(gt) 
        precision: tp / len(retrieved)
    """
    set_gt = set(gt)
    set_ret = set(retrieved)
    tp = len(set_gt & set_ret)
    recall = tp / len(gt) if gt else None
    precision = tp / len(retrieved) if retrieved else None
    return tp, recall, precision

def main(train_claims_path: str, top100_path: str):
    with open(train_claims_path, 'r', encoding='utf-8') as f:
        train_claims = json.load(f)
    with open(top100_path, 'r', encoding='utf-8') as f:
        top100 = json.load(f)

    recalls = []
    precisions = []

    for claim_id, claim_info in train_claims.items():
        gt_list = claim_info.get("evidences", [])
        retrieved_list = top100.get(claim_id, {}).get("evidences", [])

        tp, recall, precision = compute_metrics(gt_list, retrieved_list)
        recalls.append(recall if recall is not None else 0.0)
        precisions.append(precision if precision is not None else 0.0)

        # print(f"{claim_id}: hit {tp}/{len(gt_list)} → Recall@100={recall:.3f} Precision@100={precision:.3f}")

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
    print("\n=== Overall ===")
    print(f"Average Recall@100   : {avg_recall:.3f}")
    print(f"Average Precision@100: {avg_precision:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BM25 topk Recall Precision F-score"
    )
    parser.add_argument(
        "--train_claims",
        type=str,
        default="./data/dev-claims.json",
    )
    parser.add_argument(
        "--top100",
        type=str,
        default="./data/dev-claims-top100.json",
    )
    args = parser.parse_args()
    main(args.train_claims, args.top100)