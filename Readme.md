Mini Search & Ranking System

An end-to-end mini product search engine implementing retrieval, learning-to-rank, click feedback, offline evaluation, and performance benchmarking.

Built as part of a search / ranking system take-home assignment.

 Features

Synthetic data generation (50k products, 100k queries)

BM25 keyword retriever

LightGBM learning-to-rank model

Click simulation with position bias

Offline evaluation (NDCG@10)

Online-ish feedback loop

REST APIs using FastAPI

Real-time query metrics

Performance benchmarking under concurrency

Design doc and scaling strategy

 Architecture Overview
Client
  |
  v
FastAPI (/search)
  |
  v
BM25 Retriever (Top 50)
  |
  v
Feature Builder
  |
  v
LightGBM Ranker
  |
  v
Final Ranked Results

 Project Structure
mini_search_system/
│
├── api/                    # FastAPI app
├── benchmarks/            # Load testing scripts
├── data/
│   ├── raw/               # items.parquet, queries.parquet
│   └── processed/         # clicks, training data
├── retrieval/             # BM25 index + searcher
├── ranking/               # Feature engineering + ranker
├── simulation/            # Click simulator
├── training/              # Training pipeline
├── evaluation/            # Offline evaluation
├── models/                # Saved LightGBM model
├── README.md
└── requirements.txt

 Setup
1. Install dependencies
pip install -r requirements.txt
pip install httpx psutil uvicorn

 Data Generation
python data_generation/generate_items.py
python data_generation/generate_queries.py

 Build BM25 Index
python retrieval/bm25_index.py

 Simulate Clicks
python simulation/click_simulator.py

 Train Ranker
python training/prepare_training_data.py
python training/train_ranker.py

 Offline Evaluation
python evaluation/offline_eval.py


Example:

Retriever-only NDCG@10: 0.42
Retriever+Ranker NDCG@10: 0.57

 Run API Server
python -m uvicorn api.app:app --reload


Open:

http://127.0.0.1:8000/docs

 API Endpoints
Search
GET /search?q=<query>&k=20

Bulk Item Ingestion
POST /items/bulk

Click Feedback
POST /feedback/click


Body:

{
  "user_id": "u1",
  "query": "demo phone",
  "item_id": 123,
  "position": 0
}

Top Queries
GET /top_queries?window=5m

 Benchmarking
Reduce index size
python benchmarks/reduce_index.py 10000


Restart API, then:

python benchmarks/benchmark_search.py


Metrics collected:

Requests/sec

Mean / P95 / P99 latency

CPU / Memory

System saturates at ~40–45 RPS on single-process FastAPI.

 Sample Benchmark (10k Items)
Concurrency	QPS	Mean Latency	P95
50	45	0.87s	1.20s
100	40	1.56s	2.55s
200	39	3.28s	5.51s
400	39	6.22s	10.93s
 Modeling

Retriever: BM25

Ranker: LightGBM

Features:

BM25 score

Price

Query length

Title length

Final score:

0.7 × LightGBM + 0.3 × BM25

 Position Bias Handling

Click simulation uses:

P(click) ∝ 1 / log2(position + 2)


Negative samples generated from non-clicked results.

 Scaling Plan

To support 10× data and QPS:

FAISS / ANN embeddings

Async ranking

Batch inference

Redis caching

Horizontal FastAPI workers

GPU acceleration

 Author

Tushar Goswami

MS Data Science

 Status

Complete end-to-end mini search engine with retrieval, ranking, learning, evaluation, APIs, and benchmarking.