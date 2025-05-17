#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fact_checker.py
--------------
离线 MiniLM + soft‑prompt + BM25/TF‑IDF 检索 + LR 分类器
另含 Top‑K 检索结果导出与评估 (Recall@K / Precision@K)
"""

import os, json, argparse, pickle, pathlib, sys, warnings, tarfile
import numpy as np, joblib, tqdm, torch, nltk, re
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

# ============ 常量区 ============
DATA_DIR   = "COMP90042_2025/data"
EVI_PATH   = f"{DATA_DIR}/evidence.json"

LOCAL_MODEL_PATH = "./local_model"        # 本地 MiniLM 文件夹
MODEL_NAME       = "all-MiniLM-L6-v2"     # 仅作缓存回退名
BATCH_SIZE       = 64
MAX_EVI          = 5      # 传给分类器的 evidence 数
KBM25            = 200    # 初筛候选数
TOPK_DEFAULT     = 100    # dump / eval 默认 K
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP  = {"SUPPORTS":0, "REFUTES":1, "NOT_ENOUGH_INFO":2, "DISPUTED":3}
ID2LABEL   = {v:k for k,v in LABEL_MAP.items()}
# ================================

# --------- 环境准备：NLTK 资源 ---------
for res in ("punkt","stopwords","punkt_tab"):
    try:
        path = (f"tokenizers/{res}" if res.startswith("punkt") else f"corpora/{res}")
        nltk.data.find(path)
    except LookupError:
        nltk.download(res, quiet=True)

# --------- 句向量编码器加载 ----------
def build_sentence_encoder():
    print(f"🔍 Loading Sentence‑Transformer on {DEVICE} …")
    try:
        if os.path.isdir(LOCAL_MODEL_PATH):
            print(f" → from local dir: {LOCAL_MODEL_PATH}")
            return SentenceTransformer(LOCAL_MODEL_PATH, device=DEVICE, trust_remote_code=True)
        print(" → local dir missing, trying HF cache …")
        return SentenceTransformer(MODEL_NAME, device=DEVICE, trust_remote_code=True)
    except Exception as e:
        sys.exit(f"❌ Failed to load encoder: {e}\n请检查 {LOCAL_MODEL_PATH} 是否含模型文件")

# --------- 简易 tokenizer (避免 punkt 依赖过重，可替换) ---------
_word_pat = re.compile(r"\b\w+\b")
def quick_tokenize(text:str):
    return _word_pat.findall(text.lower())

# ------------------ Retriever ------------------
class Retriever:
    def __init__(self, mode="bm25", rebuild=False, soft_prompt=""):
        assert mode in ("bm25","tfidf")
        self.mode        = mode
        self.soft_prompt = soft_prompt

        self.evidence = json.load(open(EVI_PATH))
        self.ids, self.texts = zip(*self.evidence.items())
        self.idx_pkl = f"retr_{mode}.pkl"

        if rebuild or not pathlib.Path(self.idx_pkl).exists():
            self._build_index()
        else:
            self._load_index()

        # 语义重排
        self.encoder = build_sentence_encoder()
        if rebuild or not pathlib.Path("emb.npy").exists():
            self._encode_evidence()
        else:
            self._load_embeddings()

    # ---------- 索引 ----------
    def _build_index(self):
        print(f"[Retriever] build {self.mode} index")
        if self.mode == "bm25":
            tokenised = [quick_tokenize(t) for t in self.texts]
            self.bm25 = BM25Okapi(tokenised)
            joblib.dump({"bm25": self.bm25}, self.idx_pkl)
        else:
            self.vectorizer = TfidfVectorizer(stop_words="english",
                                              ngram_range=(1,2),
                                              max_df=0.9, min_df=2)
            self.evi_mat = self.vectorizer.fit_transform(self.texts)
            joblib.dump({"vec": self.vectorizer, "mat": self.evi_mat}, self.idx_pkl)

    def _load_index(self):
        obj = joblib.load(self.idx_pkl)
        if "bm25" in obj:
            self.bm25 = obj["bm25"]
        else:
            self.vectorizer, self.evi_mat = obj["vec"], obj["mat"]

    # ---------- evidence 向量 ----------
    def _encode_evidence(self):
        print("[Retriever] encode evidence embeddings …")
        embs = self.encoder.encode(self.texts, batch_size=BATCH_SIZE,
                                   normalize_embeddings=True,
                                   show_progress_bar=True)
        np.save("emb.npy", embs)
        pickle.dump(self.ids, open("ids.pkl","wb"))
        self.emb_dict = dict(zip(self.ids, embs))

    def _load_embeddings(self):
        self.ids      = pickle.load(open("ids.pkl","rb"))
        self.emb_dict = dict(zip(self.ids, np.load("emb.npy")))

    # ---------- 查询 ----------
    def retrieve(self, claim:str, k:int=MAX_EVI, return_scores=False):
        claim_aug = f"{self.soft_prompt} {claim}".strip()

        # 第一阶段
        if self.mode=="bm25":
            scores = self.bm25.get_scores(quick_tokenize(claim_aug))
        else:
            qv = self.vectorizer.transform([claim_aug])
            scores = (qv @ self.evi_mat.T).toarray()[0]
        top_idx = np.argsort(scores)[::-1][:KBM25]

        # 语义重排
        qvec = self.encoder.encode(claim_aug, normalize_embeddings=True)
        reranked = [(self.ids[i], float(qvec @ self.emb_dict[self.ids[i]]))
                    for i in top_idx]
        reranked.sort(key=lambda x:x[1], reverse=True)
        top = reranked[:k]

        if return_scores:
            return top                      # list[(eid,score)]
        return [eid for eid,_ in top], [self.evidence[eid] for eid,_ in top]

# ------------------ 分类器训练 ------------------
def train_classifier(retr_mode, soft_prompt):
    retr    = Retriever(retr_mode, soft_prompt=soft_prompt)
    encoder = build_sentence_encoder()

    def build(split):
        claims = json.load(open(f"{DATA_DIR}/{split}-claims.json"))
        X, y = [], []
        for c in tqdm.tqdm(claims.values(), desc=f"encode {split}"):
            ev = [retr.evidence[eid] for eid in c["evidences"][:MAX_EVI]]
            text = f"{soft_prompt} {c['claim_text']} </s> " + " ".join(ev)
            X.append(text.strip())
            y.append(LABEL_MAP[c["claim_label"]])
        Xemb = encoder.encode(X, batch_size=BATCH_SIZE, normalize_embeddings=False,
                              show_progress_bar=True)
        return Xemb, np.array(y)

    X, y = build("train")
    clf  = LogisticRegression(max_iter=1000, multi_class="ovr")
    clf.fit(X, y)
    joblib.dump({"clf":clf, "soft_prompt":soft_prompt, "retr_mode":retr_mode},
                "logreg_mini.joblib")
    print(f"[Classifier] saved Train acc={clf.score(X,y):.4f}")

# ------------------ 预测 ------------------
def run_split(claim_json, out_json, retr_mode, soft_prompt, limit=None):
    if not pathlib.Path("logreg_mini.joblib").exists():
        sys.exit("❌ 请先训练分类器 (--train)")
    pkg = joblib.load("logreg_mini.joblib")
    if soft_prompt!=pkg["soft_prompt"]:
        warnings.warn("当前 soft‑prompt 与训练不一致！")

    retr = Retriever(retr_mode, soft_prompt=soft_prompt)
    enc  = build_sentence_encoder()
    clf  = pkg["clf"]

    claims = json.load(open(claim_json))
    if limit: claims = dict(list(claims.items())[:limit])

    results = {}
    for cid,c in tqdm.tqdm(claims.items(), desc="predict"):
        eids, etexts = retr.retrieve(c["claim_text"])
        if etexts:
            vec = enc.encode(f"{soft_prompt} {c['claim_text']} </s> "
                             + " ".join(etexts),
                             normalize_embeddings=False)
            label = ID2LABEL[int(clf.predict([vec])[0])]
        else:
            label = "NOT_ENOUGH_INFO"
        results[cid] = {"claim_label":label, "evidences":eids}
    json.dump(results, open(out_json,"w"), indent=2)
    print(f"[Output] {out_json} saved")

# ------------------ Dev 评估 ------------------
def evaluate(pred_json, gold_json):
    p, g = json.load(open(pred_json)), json.load(open(gold_json))
    f_list, acc_list = [], []
    for cid,true in g.items():
        if cid not in p: continue
        pred = p[cid]
        acc_list.append(pred["claim_label"]==true["claim_label"])
        tp = len(set(pred["evidences"]) & set(true["evidences"]))
        prec = tp/len(pred["evidences"]) if pred["evidences"] else 0
        rec  = tp/len(true["evidences"])
        f    = 0 if prec+rec==0 else 2*prec*rec/(prec+rec)
        f_list.append(f)
    F, A = np.mean(f_list), np.mean(acc_list)
    H    = 0 if F+A==0 else 2*F*A/(F+A)
    print(f"\n[Dev] Retrieval‑F={F:.4f}  Acc={A:.4f}  Harmonic={H:.4f}\n")

# ------------------ Top‑K dump & 评估 ------------------
def dump_topk(split_json, out_json, retr_mode, soft_prompt, topk):
    retr   = Retriever(retr_mode, soft_prompt=soft_prompt)
    claims = json.load(open(split_json))
    out = {}
    for cid,c in tqdm.tqdm(claims.items(), desc=f"Top{topk}"):
        eids_scores = retr.retrieve(c["claim_text"], k=topk, return_scores=True)
        out[cid] = {"evidences":[eid for eid,_ in eids_scores]}
    json.dump(out, open(out_json,"w"), indent=2)
    print(f"[TopK] {out_json} saved")

def eval_topk(train_claims_path, topk_json):
    train_claims = json.load(open(train_claims_path))
    topk_pred    = json.load(open(topk_json))

    recalls, precisions = [], []
    for cid,c in train_claims.items():
        gt  = set(c.get("evidences", []))
        ret = set(topk_pred.get(cid, {}).get("evidences", []))
        tp  = len(gt & ret)
        recalls.append(tp/len(gt) if gt else 0)
        precisions.append(tp/len(ret) if ret else 0)

    R = sum(recalls) / len(recalls)
    P = sum(precisions) / len(precisions)
    k_val = len(next(iter(topk_pred.values()))["evidences"])  
    print(f"\n=== Retrieval Top{k_val} ===")
    print(f"Average Recall   : {R:.3f}")
    print(f"Average Precision: {P:.3f}\n")


# ------------------ CLI ------------------
def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--train",         action="store_true")
    ap.add_argument("--dev-eval",      action="store_true")
    ap.add_argument("--predict-test",  action="store_true")
    ap.add_argument("--dump-topk",     action="store_true",
                    help="导出 <split>-claims-topK.json")
    ap.add_argument("--eval-topk",     action="store_true",
                    help="计算 TopK 检索 Recall/Precision")
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT,
                    help=f"K for TopK retrieval (default {TOPK_DEFAULT})")
    ap.add_argument("--split", choices=["train","dev","test"], default="dev",
                    help="split 选择 (dump/eval 用)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--retrieval", choices=["bm25","tfidf"], default="bm25")
    ap.add_argument("--soft-prompt", default="")
    return ap.parse_args()

# ------------------ 主入口 ------------------
if __name__=="__main__":
    # 离线模式环境变量（若要联网加载可注释掉）
    os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"

    args = cli()

    # 如需重建索引 / emb
    Retriever(mode=args.retrieval, rebuild=args.rebuild_index,
              soft_prompt=args.soft_prompt)

    # ---------- Top‑K 功能 ----------
    if args.dump_topk:
        split_file = (f"{DATA_DIR}/{args.split}-claims.json" if args.split!="test"
                      else f"{DATA_DIR}/test-claims-unlabelled.json")
        out_file = f"{args.split}-claims-top{args.topk}.json"
        dump_topk(split_file, out_file,
                  args.retrieval, args.soft_prompt, args.topk)
    if args.eval_topk:
        if args.split=="test":
            sys.exit("❌ test split 无标签，无法评估")
        top_file = f"{args.split}-claims-top{args.topk}.json"
        if not pathlib.Path(top_file).exists():
            sys.exit(f"❌ {top_file} 不存在，请先 --dump-topk")
        eval_topk(f"{DATA_DIR}/{args.split}-claims.json", top_file)

    # ---------- 原有流水线 ----------
    if args.train:
        train_classifier(args.retrieval, soft_prompt=args.soft_prompt)
    if args.dev_eval:
        run_split(f"{DATA_DIR}/dev-claims.json", "dev-preds.json",
                  args.retrieval, args.soft_prompt, args.limit)
        evaluate("dev-preds.json", f"{DATA_DIR}/dev-claims.json")
    if args.predict_test:
        run_split(f"{DATA_DIR}/test-claims-unlabelled.json",
                  "test-claims-predictions.json",
                  args.retrieval, args.soft_prompt, args.limit)
        print("✅ test-claims-predictions.json ready")

# ------------------ Quick Usage Cheat‑Sheet ------------------
# 1) 建索引 + 训练 + dev 评估 + 生成 test 结果
# python fact_checker.py --rebuild-index --train --dev-eval --predict-test \
#                        --soft-prompt "[CLS] Fact‑check the following climate statement:"
#
# 2) Dump dev split Top‑100 检索结果
# python fact_checker.py --dump-topk --topk 100 --split dev
#
# 3) 评估 Recall@100 / Precision@100
# python fact_checker.py --eval-topk --topk 100 --split dev
#
# 4) 只预测 test（复用已训练分类器）
# python fact_checker.py --predict-test --soft-prompt "[CLS] Fact‑check the following climate statement:"
