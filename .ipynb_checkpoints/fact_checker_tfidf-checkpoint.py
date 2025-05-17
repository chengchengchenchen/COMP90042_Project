#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fact_checker_tfidf.py
---------------------
COMP90042 2025 – 纯 TF-IDF 检索 + MiniLM 语义重排 + LR 分类
功能：
  • --rebuild-index   构建 TF-IDF 索引 / evidence embedding
  • --train           训练分类器
  • --dev-eval        在 dev split 上预测并评估
  • --predict-test    生成 test-claims-predictions.json
  • --dump-topk       导出 Top-K evidence（train/dev/test）
  • --eval-topk       计算平均 Recall@K / Precision@K（仅 train/dev）

示例：
  # 构建索引 + 训练 + Dev 评估
  python fact_checker_tfidf.py --rebuild-index --train --dev-eval
  
  # 生成 Test 结果
  python fact_checker_tfidf.py --predict-test
  
  # Top-100 检索分析
  python fact_checker_tfidf.py --dump-topk --topk 100 --split dev
  python fact_checker_tfidf.py --eval-topk --topk 100 --split dev
"""

# ========= 依赖 =========
import os, json, argparse, pickle, pathlib, sys, warnings, re
import numpy as np, joblib, tqdm, torch, nltk
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
# ========================

# ---------- 常量 ----------
DATA_DIR   = "COMP90042_2025/data"
EVI_PATH   = f"{DATA_DIR}/evidence.json"

LOCAL_MODEL_PATH = "./local_model"             # 本地 MiniLM（推荐离线放置）
MODEL_NAME       = "all-MiniLM-L6-v2"          # HF 名称（离线失败时用）
BATCH_SIZE       = 64
MAX_EVI_RET      = 5        # 传入分类器的 evidence 数
TOPK_CANDIDATE   = 200      # TF-IDF 初筛候选数
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {"SUPPORTS":0, "REFUTES":1, "NOT_ENOUGH_INFO":2, "DISPUTED":3}
ID2LABEL  = {v:k for k,v in LABEL_MAP.items()}
# ==========================

# ---------- NLTK 资源 ----------
for res in ("punkt","stopwords","punkt_tab"):
    try:
        p = f"tokenizers/{res}" if res.startswith("punkt") else f"corpora/{res}"
        nltk.data.find(p)
    except LookupError:
        nltk.download(res, quiet=True)

# ---------- 简易 tokenizer ----------
_word_pat = re.compile(r"\b\w+\b")
def quick_tokenize(txt:str):
    return _word_pat.findall(txt.lower())

# ---------- Sentence-BERT ----------
def build_encoder():
    print(f"🔍 Loading MiniLM on {DEVICE} …")
    try:
        if os.path.isdir(LOCAL_MODEL_PATH):
            print(f" → local dir: {LOCAL_MODEL_PATH}")
            return SentenceTransformer(LOCAL_MODEL_PATH, device=DEVICE, trust_remote_code=True)
        print(" → fall back to HF cache")
        return SentenceTransformer(MODEL_NAME, device=DEVICE, trust_remote_code=True)
    except Exception as e:
        sys.exit(f"❌ Encoder load failed: {e}")

# ------------------------------------------------------------
#                     RETRIEVER – TF-IDF
# ------------------------------------------------------------
class RetrieverTFIDF:
    IDX_PKL = "retr_tfidf.pkl"

    def __init__(self, rebuild=False, soft_prompt=""):
        self.soft_prompt = soft_prompt

        self.evidence = json.load(open(EVI_PATH))
        self.ids, self.texts = zip(*self.evidence.items())

        if rebuild or not pathlib.Path(self.IDX_PKL).exists():
            self._build_index()
        else:
            self._load_index()

        # Encoding for semantic re-ranking
        self.encoder = build_encoder()
        if rebuild or not pathlib.Path("tfidf_emb.npy").exists():
            self._encode_evidence()
        else:
            self._load_embeddings()

    # ---------- Index ----------
    def _build_index(self):
        print("[Retriever] build TF-IDF index")
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1,2),
            max_df=0.9,
            min_df=2,
            dtype=np.float32
        )
        self.evi_mat = self.vectorizer.fit_transform(self.texts)
        joblib.dump({"vec": self.vectorizer, "mat": self.evi_mat}, self.IDX_PKL)

    def _load_index(self):
        obj = joblib.load(self.IDX_PKL)
        self.vectorizer, self.evi_mat = obj["vec"], obj["mat"]

    # ---------- Evidence embedding ----------
    def _encode_evidence(self):
        print("[Retriever] encode evidence embeddings …")
        embs = self.encoder.encode(
            self.texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        np.save("tfidf_emb.npy", embs)
        pickle.dump(self.ids, open("tfidf_ids.pkl","wb"))
        self.emb_dict = dict(zip(self.ids, embs))

    def _load_embeddings(self):
        self.ids      = pickle.load(open("tfidf_ids.pkl","rb"))
        self.emb_dict = dict(zip(self.ids, np.load("tfidf_emb.npy")))

    # ---------- Retrieve ----------
    def retrieve(self, claim:str, k:int=MAX_EVI_RET, return_scores=False):
        query = f"{self.soft_prompt} {claim}".strip()

        # ① TF-IDF coarse rank
        q_vec  = self.vectorizer.transform([query])
        scores = (q_vec @ self.evi_mat.T).toarray()[0]
        idx    = np.argsort(scores)[::-1][:TOPK_CANDIDATE]

        # ② MiniLM re-rank
        q_emb = self.encoder.encode(query, normalize_embeddings=True)
        reranked = [(self.ids[i], float(q_emb @ self.emb_dict[self.ids[i]]))
                    for i in idx]
        reranked.sort(key=lambda x: x[1], reverse=True)
        top = reranked[:k]

        if return_scores:
            return top
        eids = [eid for eid,_ in top]
        etxt = [self.evidence[eid] for eid in eids]
        return eids, etxt

# ------------------------------------------------------------
#                    训练 / 预测 / 评估
# ------------------------------------------------------------
def train_classifier(soft_prompt):
    retr    = RetrieverTFIDF(rebuild=False, soft_prompt=soft_prompt)
    encoder = build_encoder()

    def encode_split(split):
        claims = json.load(open(f"{DATA_DIR}/{split}-claims.json"))
        X, y = [], []
        for c in tqdm.tqdm(claims.values(), desc=f"encode {split}"):
            ev = [retr.evidence[eid] for eid in c["evidences"][:MAX_EVI_RET]]
            txt = f"{soft_prompt} {c['claim_text']} </s> " + " ".join(ev)
            X.append(txt.strip())
            y.append(LABEL_MAP[c["claim_label"]])
        Xemb = encoder.encode(X, batch_size=BATCH_SIZE,
                              normalize_embeddings=False,
                              show_progress_bar=True)
        return Xemb, np.array(y)

    X, y = encode_split("train")
    clf  = LogisticRegression(max_iter=1000, multi_class="ovr")
    clf.fit(X, y)
    joblib.dump({"clf":clf, "soft_prompt":soft_prompt}, "logreg_tfidf.joblib")
    print(f"[Classifier] saved – Train acc={clf.score(X,y):.4f}")

def run_split(claim_json, out_json, soft_prompt, limit=None):
    if not pathlib.Path("logreg_tfidf.joblib").exists():
        sys.exit("❌ 请先训练分类器 (--train)")
    clf_pkg = joblib.load("logreg_tfidf.joblib")
    if soft_prompt != clf_pkg["soft_prompt"]:
        warnings.warn("当前 soft-prompt 与训练时不一致！")

    retr = RetrieverTFIDF(rebuild=False, soft_prompt=soft_prompt)
    enc  = build_encoder()
    clf  = clf_pkg["clf"]

    claims = json.load(open(claim_json))
    if limit:
        claims = dict(list(claims.items())[:limit])

    res = {}
    for cid,c in tqdm.tqdm(claims.items(), desc="predict"):
        eids, etexts = retr.retrieve(c["claim_text"])
        if etexts:
            vec = enc.encode(f"{soft_prompt} {c['claim_text']} </s> "
                             + " ".join(etexts),
                             normalize_embeddings=False)
            label = ID2LABEL[int(clf.predict([vec])[0])]
        else:
            label = "NOT_ENOUGH_INFO"
        res[cid] = {"claim_label":label, "evidences":eids}
    json.dump(res, open(out_json,"w"), indent=2)
    print(f"[Output] {out_json} saved")

def evaluate(pred_json, gold_json):
    p, g = json.load(open(pred_json)), json.load(open(gold_json))
    f_list, acc_list = [], []
    for cid,true in g.items():
        if cid not in p: continue
        pred = p[cid]
        acc_list.append(pred["claim_label"] == true["claim_label"])
        tp = len(set(pred["evidences"]) & set(true["evidences"]))
        prec = tp/len(pred["evidences"]) if pred["evidences"] else 0
        rec  = tp/len(true["evidences"])
        f    = 0 if prec+rec==0 else 2*prec*rec/(prec+rec)
        f_list.append(f)
    F, A = np.mean(f_list), np.mean(acc_list)
    H    = 0 if F+A==0 else 2*F*A/(F+A)
    print(f"\nDev  Retrieval-F={F:.4f}  Acc={A:.4f}  Harmonic={H:.4f}\n")

# ---------- Top-K 导出 ----------
def dump_topk(split_json, out_json, soft_prompt, k):
    retr = RetrieverTFIDF(rebuild=False, soft_prompt=soft_prompt)
    claims = json.load(open(split_json))
    out={}
    for cid,c in tqdm.tqdm(claims.items(), desc=f"Top{k}"):
        eids_scores = retr.retrieve(c["claim_text"], k=k, return_scores=True)
        out[cid] = {"evidences":[eid for eid,_ in eids_scores]}
    json.dump(out, open(out_json,"w"), indent=2)
    print(f"[TopK] {out_json} saved")

# ---------- Top-K 评估 ----------
def eval_topk(train_claims_path: str, topk_json: str):
    train_claims = json.load(open(train_claims_path))
    topk_pred    = json.load(open(topk_json))

    recalls, precisions = [], []
    for cid, c in train_claims.items():
        gt = set(c.get("evidences", []))
        ret = set(topk_pred.get(cid, {}).get("evidences", []))
        if not gt:
            continue
        tp  = len(gt & ret)
        recalls.append(tp / len(gt))
        precisions.append(tp / len(ret) if ret else 0)

    R = sum(recalls) / len(recalls)
    P = sum(precisions) / len(precisions)
    k_val = len(next(iter(topk_pred.values()))["evidences"])

    print(f"\n=== Retrieval Top{k_val} ===")
    print(f"Average Recall   : {R:.3f}")
    print(f"Average Precision: {P:.3f}\n")

# ------------------------------------------------------------
#                        CLI
# ------------------------------------------------------------
def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--train",         action="store_true")
    ap.add_argument("--dev-eval",      action="store_true")
    ap.add_argument("--predict-test",  action="store_true")

    # Top-K
    ap.add_argument("--dump-topk", action="store_true")
    ap.add_argument("--eval-topk", action="store_true")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--split", choices=["train","dev","test"], default="dev")

    ap.add_argument("--limit", type=int)
    ap.add_argument("--soft-prompt", default="")
    return ap.parse_args()

# ------------------------------------------------------------
#                       主入口
# ------------------------------------------------------------
if __name__ == "__main__":
    # 离线模式（若在线可注释）
    os.environ["HF_HUB_OFFLINE"]       = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args = cli()

    # (可选) 重建索引 / embedding
    if args.rebuild_index:
        RetrieverTFIDF(rebuild=True, soft_prompt=args.soft_prompt)

    # ------ Top-K 导出 ------
    if args.dump_topk:
        split_file = (f"{DATA_DIR}/{args.split}-claims.json"
                      if args.split!="test"
                      else f"{DATA_DIR}/test-claims-unlabelled.json")
        out_file = f"{args.split}-claims-top{args.topk}.json"
        dump_topk(split_file, out_file, args.soft_prompt, args.topk)

    # ------ Top-K 评估 ------
    if args.eval_topk:
        if args.split == "test":
            sys.exit("❌ test split 无标签，无法评估 Top-K")
        top_file = f"{args.split}-claims-top{args.topk}.json"
        if not pathlib.Path(top_file).exists():
            sys.exit(f"❌ {top_file} 不存在，请先 --dump-topk")
        eval_topk(f"{DATA_DIR}/{args.split}-claims.json", top_file)

    # ------ 训练 ------
    if args.train:
        train_classifier(args.soft_prompt)

    # ------ Dev Eval ------
    if args.dev_eval:
        run_split(f"{DATA_DIR}/dev-claims.json", "dev-preds.json",
                  args.soft_prompt, args.limit)
        evaluate("dev-preds.json", f"{DATA_DIR}/dev-claims.json")

    # ------ Test 预测 ------
    if args.predict_test:
        run_split(f"{DATA_DIR}/test-claims-unlabelled.json",
                  "test-claims-predictions.json",
                  args.soft_prompt, args.limit)
        print("✅ test-claims-predictions.json ready")
