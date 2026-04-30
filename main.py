from pathlib import Path

from src.data_loader import load_corpus
from src.models import MODEL_NAMES
from src.train import run_c1, run_c2
from src.evaluate import macro_average

DATA_DIR = Path('ToS')
SEP = "=" * 62


def _print(label: str, results: list) -> None:
    if not results:
        print(f"  {label:<26}  no results")
        return
    p, r, f1 = macro_average(results)
    print(f"  {label:<26}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  (docs={len(results)})")


def main() -> None:
    print("Loading corpus...")
    docs = load_corpus(DATA_DIR)
    total_sents = sum(len(d['sentences']) for d in docs)
    total_pos = sum(lbl for d in docs for lbl in d['labels'] if lbl == 1)
    print(f"  {len(docs)} documents | {total_sents} sentences | "
          f"{total_pos} positive ({total_pos / total_sents * 100:.1f}%)")
    print(f"  Evaluation: 5-fold document-level cross-validation\n")

    # ── C1: single classifier ─────────────────────────────────────────────────
    print(SEP)
    print("C1  Single classifier  (general unfairness label)")
    print(SEP)
    for name in MODEL_NAMES:
        print(f"  Running {name}...", end='\r')
        _print(name, run_c1(docs, name))

    # ── C2: 8 category classifiers OR-combined ────────────────────────────────
    print(f"\n{SEP}")
    print("C2  8 category classifiers  (OR-combined)")
    print(SEP)
    for name in MODEL_NAMES:
        print(f"  Running {name}...", end='\r')
        _print(name, run_c2(docs, name))

    print()


if __name__ == '__main__':
    main()
