"""
Embedding Experiment: Compare TF-IDF cosine vs SBERT (all-MiniLM-L6-v2)
on retrieval quality over the ToS corpus.

Evaluation:
  - 80/20 document split (40 train, 10 test, seed=42)
  - Build index on train sentences
  - Query with EVERY test sentence that has label=1 (unfair)
  - Metrics (averaged over query sentences):
      P@5   Precision at 5
      P@10  Precision at 10
      MAP   Mean Average Precision

Usage:
  python scripts/embedding_experiment.py [--data-dir ToS]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_corpus
from service.embeddings import get_embedder
from service.corpus_index import CorpusIndex


def precision_at_k(retrieved_labels: list[bool], k: int) -> float:
    """Fraction of top-k retrieved that are unfair."""
    top = retrieved_labels[:k]
    if not top:
        return 0.0
    return sum(top) / len(top)


def average_precision(retrieved_labels: list[bool]) -> float:
    """Area under the precision-recall curve for a single query."""
    hits = 0
    ap = 0.0
    for i, rel in enumerate(retrieved_labels, start=1):
        if rel:
            hits += 1
            ap += hits / i
    return ap / hits if hits > 0 else 0.0


def run_experiment(docs: list[dict], train_docs: list[dict], test_docs: list[dict],
                   method: str) -> dict:
    embedder = get_embedder(method)
    index = CorpusIndex(embedder)
    index.build(train_docs)

    p5_scores, p10_scores, map_scores = [], [], []
    n_queries = 0

    for doc in test_docs:
        for sent, lbl in zip(doc['sentences'], doc['labels']):
            if lbl != 1:
                continue  # only use unfair sentences as queries
            n_queries += 1
            results = index.search(sent, top_k=10)
            labels = [r.is_unfair for r in results]
            p5_scores.append(precision_at_k(labels, 5))
            p10_scores.append(precision_at_k(labels, 10))
            map_scores.append(average_precision(labels))

    return {
        "method": method,
        "n_queries": n_queries,
        "P@5": np.mean(p5_scores) if p5_scores else 0.0,
        "P@10": np.mean(p10_scores) if p10_scores else 0.0,
        "MAP": np.mean(map_scores) if map_scores else 0.0,
    }


def print_table(results: list[dict]) -> None:
    header = f"{'Method':<8}  {'N Queries':>10}  {'P@5':>8}  {'P@10':>8}  {'MAP':>8}"
    sep = "-" * len(header)
    print("\n" + sep)
    print("Embedding Method Comparison — Retrieval Quality")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(f"{r['method']:<8}  {r['n_queries']:>10}  {r['P@5']:>8.4f}  "
              f"{r['P@10']:>8.4f}  {r['MAP']:>8.4f}")
    print(sep + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare embedding methods on retrieval quality.")
    parser.add_argument("--data-dir", default="ToS", help="Path to ToS data directory")
    parser.add_argument("--methods", nargs="+", default=["tfidf", "sbert"],
                        help="Embedding methods to compare")
    args = parser.parse_args()

    print(f"Loading corpus from {args.data_dir}...")
    docs = load_corpus(Path(args.data_dir))
    print(f"  {len(docs)} documents loaded.")

    # 80/20 split by document (deterministic)
    rng = random.Random(42)
    shuffled = docs[:]
    rng.shuffle(shuffled)
    split = int(0.8 * len(shuffled))
    train_docs = shuffled[:split]
    test_docs = shuffled[split:]
    print(f"  Train: {len(train_docs)} docs | Test: {len(test_docs)} docs\n")

    results = []
    for method in args.methods:
        print(f"Running experiment: {method}...")
        r = run_experiment(docs, train_docs, test_docs, method)
        results.append(r)
        print(f"  {method}: P@5={r['P@5']:.4f}  P@10={r['P@10']:.4f}  MAP={r['MAP']:.4f}")

    print_table(results)

    winner = max(results, key=lambda x: x["MAP"])
    print(f"Best embedding method by MAP: {winner['method'].upper()} "
          f"(MAP={winner['MAP']:.4f})\n")


if __name__ == "__main__":
    main()
