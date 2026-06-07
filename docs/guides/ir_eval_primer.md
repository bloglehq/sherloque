# IR Evaluation Primer: A Beginner's Guide to Measuring Search Quality

*Written for the sherloque project — a learning search engine with BM25, vector retrieval, and RRF fusion.*

---

## Part 1 — The Core Problem of IR Evaluation

### What Does "Evaluate" Even Mean?

When you build a web app, you can write unit tests: does this function return the right value? Evaluation for a search engine is harder, because "right" is fuzzy. If I search for "does aspirin reduce fever?", is this document relevant?

> *"Aspirin is a widely used analgesic and antipyretic drug..."*

Maybe yes. What about this one?

> *"NSAIDs, of which aspirin is a subclass, have been shown to reduce core body temperature in febrile patients."*

Almost certainly yes — but it requires more knowledge to judge. "Relevance" is a human judgment. It depends on the query, the document, and what the searcher actually needed.

This is the foundational challenge of IR (Information Retrieval) evaluation: **we need human judgments as ground truth, but humans are expensive, slow, and sometimes disagree.** The entire evaluation infrastructure you'll encounter is built around collecting those judgments once, carefully, and then using them to score many systems.

### Qrels: The Ground Truth

**Qrels** (short for *query relevance judgments*) are the ground truth for evaluation. A qrels file is simple: for each `(query_id, document_id)` pair, a human annotator recorded how relevant that document is to that query.

Example qrels (tab-separated):
```
query_id    doc_id          relevance
q1          doc_42          1
q1          doc_107         0
q2          doc_55          1
```

**Binary relevance** means the label is 0 (not relevant) or 1 (relevant). Most BEIR datasets, including SciFact, use binary labels. **Graded relevance** allows scores like 0, 1, 2, 3 — capturing "somewhat relevant" vs "highly relevant." TREC and MS MARCO use graded labels.

**Where do these labels come from?** Paid annotators, crowdsourcing (Mechanical Turk), or domain experts. For SciFact specifically, the labels were created by annotators with biomedical expertise who read scientific papers and marked which ones actually supported or refuted each claim.

**A crucial property of SciFact**: most queries have only **1–2 relevant documents** in the entire corpus of ~5,000 papers. This matters enormously when you interpret your numbers later — keep it in mind.

### The "Run": What Your System Returns

A **run** is the ranked list of documents your system returns for each query. It's the output side of evaluation — your system's answer, expressed as an ordered list.

```
query_id    doc_id      score (or rank)
q1          doc_99      0.94
q1          doc_42      0.91
q1          doc_7       0.88
...
```

The evaluation loop is: for each query, take your system's ranked run, compare it to the qrels, and compute a score. Do this for all queries, then average. That's your overall metric.

---

## Part 2 — The Metrics

Let's work through each metric with a concrete example. Suppose we have one query with the following ground truth: documents **D2** and **D5** are relevant (relevance = 1). Everything else is irrelevant.

Your system returns this ranked list (rank 1 is the top result):

```
Rank 1: D9   (irrelevant)
Rank 2: D2   (RELEVANT ✓)
Rank 3: D7   (irrelevant)
Rank 4: D5   (RELEVANT ✓)
Rank 5: D1   (irrelevant)
Rank 6: D3   (irrelevant)
...
```

We'll use this running example throughout.

---

### Precision@k and Recall@k

**Precision@k** answers: *"Of the top k results I showed the user, what fraction were actually relevant?"*

```
P@k = (# relevant docs in top k) / k
```

For our example:
- P@1 = 0/1 = 0.0  (D9 is irrelevant)
- P@3 = 1/3 ≈ 0.33  (only D2 is relevant in top 3)
- P@5 = 2/5 = 0.4   (D2 and D5 are in top 5)

**Recall@k** answers: *"Of ALL the relevant documents that exist, how many did I find in my top k?"*

```
Recall@k = (# relevant docs in top k) / (total # relevant docs)
```

For our example (2 relevant docs total):
- Recall@1 = 0/2 = 0.0
- Recall@3 = 1/2 = 0.5
- Recall@5 = 2/2 = 1.0  (we found both!)

**The fundamental tradeoff**: You can trivially maximize recall by returning everything — but then precision collapses. You can maximize precision by only returning docs you're very confident about — but you'll miss relevant ones. Good retrieval balances both.

**Why Recall@100 and Precision@10 answer different questions:**

- **P@10** measures quality of the top of your list — what the user actually sees on page 1. This matters for end-user experience.
- **Recall@100** measures how much of the relevant pool you've retrieved in your top 100. This matters for **pipelines**: if you're going to pass the top 100 to a reranker, recall@100 is the ceiling on what the reranker can possibly find. If the relevant doc isn't in the top 100, no reranker can save you.

---

### MRR (Mean Reciprocal Rank) — NOT RRF

> **Important clarification**: The project uses two things with similar-sounding names that are completely different:
> - **MRR** = Mean Reciprocal Rank. An *evaluation metric*. It measures how good your results are.
> - **RRF** = Reciprocal Rank Fusion. A *fusion technique*. It combines ranked lists from multiple retrievers.
>
> MRR appears in your evaluation table (the thing you measure). RRF is part of your retrieval pipeline (the thing you build). They are not related.

**MRR** measures: *"How quickly does the user find the first relevant result?"*

For one query, the **reciprocal rank** is `1 / rank_of_first_relevant_doc`:

```
Rank 1: D9   (irrelevant)
Rank 2: D2   (RELEVANT ✓)  ← first relevant doc is at rank 2

Reciprocal Rank = 1/2 = 0.5
```

If the first relevant doc were at rank 1, RR = 1.0. At rank 5, RR = 0.2.

**Mean** Reciprocal Rank averages this over all queries:

```
MRR = (1/|Q|) * Σ (1 / rank_of_first_relevant_doc_for_query_i)
```

MRR is great for tasks where the user just wants **one good result** — like a "I'm Feeling Lucky" use case, or factoid question answering. If your system consistently surfaces a relevant document at rank 2 instead of rank 1, MRR will catch that.

---

### MAP (Mean Average Precision)

MAP is more nuanced. It tries to capture ranking quality across all recall levels — not just where the *first* relevant doc is, but how well the whole ranked list is organized.

**Step 1: Average Precision for one query.**

For each relevant document found, compute the precision at that rank. Then average those precision values.

Using our running example:
```
Rank 1: D9   irrelevant     → skip
Rank 2: D2   RELEVANT ✓     → P@2 = 1/2 = 0.50
Rank 3: D7   irrelevant     → skip
Rank 4: D5   RELEVANT ✓     → P@4 = 2/4 = 0.50
Rank 5: D1   irrelevant     → skip
...

Average Precision = (0.50 + 0.50) / 2 = 0.50
```

Notice: if D2 and D5 were both at ranks 1 and 2, AP would be:
```
P@1 = 1.0, P@2 = 1.0  → AP = (1.0 + 1.0) / 2 = 1.0
```

If they were at ranks 9 and 10:
```
P@9 = 1/9 ≈ 0.11, P@10 = 2/10 = 0.20  → AP = (0.11 + 0.20) / 2 = 0.155
```

MAP rewards you for putting relevant docs high. It penalizes you for finding relevant docs but burying them.

**Step 2: Mean over all queries.**

```
MAP = (1/|Q|) * Σ AP(query_i)
```

MAP is sometimes computed with a cutoff (MAP@100) — only considering the top 100 results per query. The ranx library does this automatically.

---

### nDCG@k (Normalized Discounted Cumulative Gain)

nDCG is the most important and commonly reported metric for ranking quality. It's the "headline number" in virtually every IR paper. Let's build up to it.

#### Gain

A document's **gain** is its relevance score. For binary relevance (SciFact), this is just 1 for relevant and 0 for irrelevant. For graded relevance with scores 0–3, the gain might be 0, 1, 2, or 3.

#### Cumulative Gain

**Cumulative Gain@k** (CG@k) just sums up the gains of the top k results:

```
CG@5 for our example:
Rank 1: D9   gain=0
Rank 2: D2   gain=1
Rank 3: D7   gain=0
Rank 4: D5   gain=1
Rank 5: D1   gain=0

CG@5 = 0 + 1 + 0 + 1 + 0 = 2
```

Problem: CG doesn't care about position. Returning D2 at rank 2 vs rank 200 gives the same CG. That's clearly wrong for a search engine.

#### Discounted Cumulative Gain (DCG)

**DCG** adds a logarithmic discount by position. Documents at lower ranks contribute less:

```
DCG@k = Σ (gain_i / log2(rank_i + 1))
```

The `log2(rank + 1)` denominator grows slowly, so:
- Rank 1: discount = log2(2) = 1.0  → no penalty
- Rank 2: discount = log2(3) ≈ 1.585
- Rank 4: discount = log2(5) ≈ 2.322
- Rank 10: discount = log2(11) ≈ 3.459

**Why the logarithm?** It models user behavior: users are much less likely to click on rank 5 than rank 1, but the drop-off slows down as you go deeper. The log captures this diminishing marginal cost of going lower.

Computing DCG@5 for our example:
```
Rank 1: D9   gain=0   → 0/log2(2)   = 0
Rank 2: D2   gain=1   → 1/log2(3)   ≈ 0.631
Rank 3: D7   gain=0   → 0/log2(4)   = 0
Rank 4: D5   gain=1   → 1/log2(5)   ≈ 0.431
Rank 5: D1   gain=0   → 0/log2(6)   = 0

DCG@5 = 0 + 0.631 + 0 + 0.431 + 0 = 1.062
```

#### Normalized DCG (nDCG)

DCG is hard to compare across queries. If a query has 10 relevant docs, its maximum possible DCG is much higher than a query with 1 relevant doc. We need to normalize.

**IDCG** (Ideal DCG) is the DCG you'd get if you returned the relevant documents in a perfect order — all relevant docs at the top.

For our example, the ideal ranking would have D2 and D5 at ranks 1 and 2:
```
Ideal:
Rank 1: D2   gain=1   → 1/log2(2)   = 1.0
Rank 2: D5   gain=1   → 1/log2(3)   ≈ 0.631
Rank 3-5: irrelevant

IDCG@5 = 1.0 + 0.631 = 1.631
```

Then:
```
nDCG@5 = DCG@5 / IDCG@5 = 1.062 / 1.631 ≈ 0.651
```

nDCG ranges from 0 to 1. A perfect ranking gets 1.0. This makes it easy to compare across queries and across systems.

**nDCG@10** is the standard because it captures ranking quality across roughly "one page" of results — the 10 documents a user sees without scrolling or clicking to page 2.

---

## Part 3 — Reading Your Actual Results

Here are your measured numbers on beir/scifact/test:

| Metric       | BM25   | Vector | RRF    |
|--------------|--------|--------|--------|
| nDCG@10      | 0.6316 | 0.7028 | 0.7053 |
| Recall@100   | 0.8652 | 0.8977 | 0.9577 |
| MAP          | 0.5940 | 0.6554 | 0.6685 |
| MRR@10       | 0.5986 | 0.6662 | 0.6744 |
| P@10         | 0.0817 | 0.0950 | 0.0920 |

Let's read this carefully.

### Why RRF Wins Most Decisively on Recall@100

RRF jumps from 0.8977 (vector alone) to **0.9577** — a gain of +0.06. That's the biggest relative improvement RRF makes anywhere in the table.

Here's the intuition: BM25 and your vector retriever fail on *different* queries and *different* documents. BM25 is good at exact keyword matches ("aspirin fever") but misses paraphrases. Vector retrieval is good at semantic similarity but can miss rare technical terms. When you fuse with RRF, you get the **union of what each retriever found**. A document that only BM25 found (missed by vector) can still make the top 100 after fusion.

Recall@100 measures exactly this: what fraction of the relevant documents did you capture anywhere in the top 100? Fusion's union-of-retrievers behavior is exactly what maximizes this.

### Why RRF Barely Moves nDCG@10 Over Vector Alone

Vector retrieval gets nDCG@10 = 0.7028. RRF gets 0.7053 — essentially the same.

This is a crucial insight. The documents that BM25 recovers that vector missed are **being added to the pool**, but they're not making it into the top 10 positions after fusion. The fusion step isn't smart enough to rerank them to the top.

What this is telling you: **you have a reranking problem, not a retrieval problem.** Your RRF pool at position 11–100 probably contains relevant documents that a good reranker (like a cross-encoder) could surface into the top 10. The gain from fusion is sitting in the pool, waiting to be extracted. We'll return to this in Part 5.

### Why P@10 is Low — and Why That's Fine

Your P@10 numbers: BM25 = 0.0817, Vector = 0.0950, RRF = 0.0920.

These look terrible. What's going on?

Remember from Part 1: most SciFact queries have **only 1–2 relevant documents total**. If a query has exactly 1 relevant document, the *maximum possible* P@10 is:

```
P@10 (best case) = 1 relevant doc / 10 results = 0.10
```

You literally cannot score above 0.10 on P@10 for that query, even with a perfect system. If most queries have 1 relevant doc and you're finding it at rank 1–3 (which is good!), you'll see P@10 ≈ 0.08–0.10.

This is a **dataset property**, not a quality problem. SciFact was designed for claim verification — each scientific claim has a small number of supporting papers. Don't be alarmed by these numbers. P@10 would be a meaningful metric if you had, say, 50+ relevant docs per query (like in MS MARCO passage retrieval).

### The Sanity Check Baseline

A common reference point from the BEIR paper and Anserini (the battle-tested Lucene-based BM25 library) is **BM25 nDCG@10 ≈ 0.665** on SciFact.

Your BM25 lands at **0.6316** — about 3 points lower. Is this a problem?

Not really, and here's why. Anserini uses Lucene's production text analyzer: Porter stemming (reduces "running" → "run"), a curated English stopword list, and battle-hardened tokenization. Your BM25 implementation uses NLTK's tokenizer and lemmatization. These produce slightly different token streams, which changes the term statistics that BM25 relies on.

A 3-point gap on BM25 alone, when your fused system reaches 0.7053, is completely reasonable for a learning project. You haven't done anything wrong — you've just confirmed that text preprocessing choices matter for lexical retrieval.

---

## Part 4 — The Dataset & Tooling Landscape

### BEIR: The Standard Benchmark

**BEIR** (Benchmarking IR) is a collection of approximately 18 diverse IR datasets, released in 2021 by Thakur et al. Its goal: test whether a retrieval system generalizes across domains without any fine-tuning on the target domain (*zero-shot* evaluation).

Before BEIR, most retrieval research optimized for a single dataset (usually MS MARCO). BEIR revealed that methods that crushed MS MARCO often failed badly on medical, legal, or scientific text. It forced the field to think about generalization.

BEIR is now the standard table you see in virtually every retrieval paper. When someone says "we evaluated on BEIR," they mean they tested on (a subset of) its 18 datasets and reported the average.

### SciFact Specifically

**SciFact** is one of the BEIR datasets. It contains:
- ~5,000 scientific paper abstracts (the corpus)
- ~300 test queries (scientific claims like "ACE2 is the receptor for SARS-CoV-2")
- Binary qrels annotated by biomedical experts

What makes it good for learning: it's **small**. Indexing 5K documents takes seconds. Running an evaluation takes a few seconds. You can iterate quickly without waiting for hours. That said, 300 queries is a reasonably robust test set — you can trust the numbers you see.

### Other Datasets You'll Encounter

**MS MARCO (Passage Ranking)**
The biggest dataset in modern retrieval. ~8.8 million passages, ~500K training queries, ~7K dev queries. This is where dense retrieval (bi-encoders, DPR, etc.) was developed and popularized. If a paper claims state-of-the-art retrieval, they almost certainly trained on MS MARCO. Useful for: training retrieval models, benchmarking at massive scale.

**TREC Tracks**
TREC (Text REtrieval Conference) is NIST's annual competition that's been running since 1992. It's the origin of formal IR evaluation — qrels, runs, and most of the metrics in this document come from TREC. "TREC 2019 DL" (Deep Learning track) is a commonly cited benchmark with graded relevance. Useful for: historical comparison, graded-relevance evaluation, very high-quality human judgments.

**Natural Questions (NQ)**
Google's open-domain QA dataset, ~300K questions from real Google searches with answers from Wikipedia. Widely used for retrieval-augmented QA (retrieve a Wikipedia passage, then answer the question). Useful for: open-domain QA evaluation, testing how well retrieval supports a downstream reader.

**HotpotQA**
Multi-hop question answering — questions that require reasoning across two or more documents. ("What country is the headquarters of the company that makes the iPhone?") Useful for: testing whether your retrieval can handle complex, compositional queries.

**FiQA**
Financial question-answering from community forums (Stack Exchange finance, Reddit). Small (~57K corpus) but domain-specific. Useful for: domain-shift testing — does your general-purpose retriever work for specialized financial language?

**TREC-COVID**
Built during the COVID-19 pandemic from biomedical literature (CORD-19). Queries are medical research questions. Useful for: biomedical retrieval, testing on a domain with very specialized vocabulary.

### ir_datasets: Uniform Dataset Access

**`ir_datasets`** is the Python library your scripts use (`import ir_datasets`). It provides a uniform interface to dozens of IR datasets — you write the same code whether you're loading SciFact, MS MARCO, or TREC-COVID.

For each dataset, it exposes:
- `dataset.docs_iter()` — iterate over documents (id + text)
- `dataset.queries_iter()` — iterate over queries (id + text)
- `dataset.qrels_iter()` — iterate over relevance judgments (query_id, doc_id, relevance)

Without `ir_datasets`, you'd be dealing with different file formats, download scripts, and data structures for every dataset. It makes switching from SciFact to another BEIR dataset a one-line change.

### ranx: Computing the Metrics

**`ranx`** is the library your eval scripts use to compute all the metrics above. It has three key concepts:

```python
from ranx import Qrels, Run, evaluate

# Qrels: wrap the ground truth
qrels = Qrels({"q1": {"doc2": 1, "doc5": 1}, ...})

# Run: wrap your system's ranked results
run = Run({"q1": {"doc9": 0.94, "doc2": 0.91, "doc7": 0.88, ...}, ...})

# evaluate: compute any metric
results = evaluate(qrels, run, ["ndcg@10", "recall@100", "map", "mrr@10", "precision@10"])
```

`ranx` handles all the math: computing per-query scores and averaging over queries. It's fast (written with Numba), supports all standard metrics, and has a clean API. Your `eval_bm25.py` and `eval_retrieval.py` scripts use exactly this pattern.

---

## Part 5 — Connecting Evaluation to Building a Real Search Engine

### The Retrieve → Fuse → Rerank Pipeline

Your current pipeline looks like this:

```
Query
  │
  ├─── BM25 (top-100) ──────────────┐
  │                                  ├─── RRF Fusion ──→ top-100 fused
  └─── Vector (top-100) ────────────┘
```

Evaluation fits in at each stage:
- Evaluate BM25 alone → your `eval_bm25.py` baseline
- Evaluate vector alone → add to `eval_retrieval.py`
- Evaluate RRF → measure how much fusion helps

This tells you *where* the pipeline gains and loses quality.

The next stage is a **reranker** — typically a cross-encoder model that takes the fused top-100 and reorders them more carefully:

```
Query + fused top-100
  │
  └─── Cross-encoder reranker ──→ reranked top-10
```

After adding a reranker, you'd evaluate nDCG@10 again. Given your current numbers (Recall@100 = 0.9577 means the relevant docs are almost certainly in the pool), a good reranker could plausibly push nDCG@10 from 0.70 toward 0.80+. That's the lever the numbers are pointing at.

### Offline vs Online Metrics

Everything in this document is an **offline metric**: computed against a fixed set of human-labeled queries, with no real users involved. Offline metrics are:
- Reproducible (same inputs → same score)
- Fast (you can evaluate in seconds)
- Cheap (no user experiment needed)

But they're not the whole story. **Online metrics** — click-through rate, session success rate, time-to-first-click — measure what real users actually do with your results. A system can score well offline but fail online if the relevant documents are written in a dry academic style that users skip past.

For a learning project, offline metrics are exactly right. Just know they exist on a spectrum: offline evaluation tells you "can the system find relevant documents?", online evaluation tells you "do users actually succeed?".

### What to Measure Next

Your results suggest a clear roadmap:

1. **Add a cross-encoder reranker** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2` from Hugging Face). Pass the fused top-100 to it, take the top-10, and re-evaluate nDCG@10. Your high Recall@100 means the relevant docs are in the pool — a good reranker should surface them.

2. **Try other BEIR datasets** — SciFact is convenient but small. Trying FiQA or TREC-COVID will tell you if your retrieval generalizes or if you've overfit your preprocessing to scientific abstracts.

3. **Ablate your text preprocessing** — try different tokenizers, stemming strategies, stopword lists. Your 0.63 vs 0.665 gap with Anserini is mostly a preprocessing story.

The evaluation infrastructure you've already built (`eval_retrieval.py` + `ranx` + `ir_datasets`) makes all of this easy. That's the real value of setting it up properly: you can measure the contribution of every change you make.

---

## Quick Reference

| Metric | What it measures | When it matters most |
|--------|-----------------|---------------------|
| P@k | Fraction of top-k that are relevant | User-facing quality, page 1 experience |
| Recall@k | Fraction of all relevant docs found in top-k | Pipeline ceiling; reranker input quality |
| MRR | Rank of first relevant result | QA / "find me one good answer" tasks |
| MAP | Ranking quality across all recall levels | When you care about the whole ranked list |
| nDCG@k | Position-discounted ranking quality | Standard headline metric; use this first |

| Acronym | Stands for | What it is |
|---------|-----------|------------|
| IR | Information Retrieval | The field |
| MRR | Mean Reciprocal Rank | An evaluation metric |
| RRF | Reciprocal Rank Fusion | A retrieval fusion technique (not a metric) |
| MAP | Mean Average Precision | An evaluation metric |
| nDCG | Normalized Discounted Cumulative Gain | An evaluation metric |
| BEIR | Benchmarking IR | A collection of ~18 IR evaluation datasets |
| IDCG | Ideal DCG | The maximum possible DCG (for normalization) |
| qrels | Query relevance judgments | The ground truth labels |
