# 顶部额外依赖
from pyserini.index import SimpleIndexer
from pyserini.search import SimpleSearcher

# 在常量区加入
BM25_INDEX = "evi_bm25"
KBM25 = 200

# ---------- Retriever._build_index 末尾再建 BM25 ----------
        # 3) 建 Pyserini 索引
        from tqdm import tqdm
        idx = SimpleIndexer(BM25_INDEX, overwrite=True)
        for eid, txt in tqdm(self.evidence.items(), desc="build BM25"):
            idx.add(docid=eid, contents=txt)
        idx.commit()

# ---------- Retriever.__init__ 里 if not rebuild & cache ----------
        # 加载 BM25
        self.searcher = SimpleSearcher(BM25_INDEX)
        self.searcher.set_bm25(k1=0.9, b=0.4)

# ---------- 把 retrieve() 改成 ↓ ----------
    def retrieve(self, claim, k=MAX_EVI):
        # ① 先 BM25 取 200
        hits = self.searcher.search(claim, k=KBM25)
        bm25_ids = [h.docid for h in hits]
        bm25_scores = {h.docid: h.score for h in hits}

        # ② MiniLM 重排
        qvec = self.encoder.encode(claim, normalize_embeddings=True)
        scored = []
        for eid in bm25_ids:
            sem = float(qvec @ self.emb_dict[eid])
            # 0‑1 归一化 bm25：除以 max
            bm = bm25_scores[eid] / hits[0].score
            score = 0.9*sem + 0.1*bm
            scored.append((eid, score))
        scored.sort(key=lambda x:x[1], reverse=True)
        top = scored[:k]
        return [eid for eid,_ in top], [self.evidence[eid] for eid,_ in top]
