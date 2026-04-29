from pathlib import Path
from src.data_loader import load_corpus

docs = load_corpus(Path('ToS'))
total_sents = sum(len(d['sentences']) for d in docs)
total_pos = sum(lbl for d in docs for lbl in d['labels'] if lbl == 1)
print(f"{len(docs)} docs, {total_sents} sentences, {total_pos} positive ({total_pos/total_sents*100:.1f}%)")

# Check category labels on first doc
d = docs[0]
print(f"Doc: {d['name']}, sentences: {len(d['sentences'])}, labels: {len(d['labels'])}")
for cat, labels in d['cat_labels'].items():
    pos = sum(1 for l in labels if l == 1)
    print(f"  {cat}: {pos} positive")
