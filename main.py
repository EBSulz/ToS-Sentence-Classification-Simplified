from pathlib import Path

from src.data_loader import load_corpus
from src.models import MODEL_NAMES
from src.train import run_c1, run_c2
from src.evaluate import macro_average

DATA_DIR = Path('ToS')


def _print_results(model_name: str, fold_results: list) -> None:
    if not fold_results:
        print(f"  {model_name:<22}  no results")
        return
    p, r, f1 = macro_average(fold_results)
    folds = len(fold_results)
    print(f"  {model_name:<22}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  (folds={folds})")


def main() -> None:
    print("Loading corpus...")
    docs = load_corpus(DATA_DIR)
    total_sents = sum(len(d['sentences']) for d in docs)
    total_pos = sum(lbl for d in docs for lbl in d['labels'] if lbl == 1)
    print(f"  {len(docs)} documents, {total_sents} sentences, {total_pos} positive ({total_pos/total_sents*100:.1f}%)")

    sep = "=" * 60

    print(f"\n{sep}")
    print("C1 — Single classifier on general unfairness label")
    print(sep)
    for model_name in MODEL_NAMES:
        print(f"  Running {model_name}...", end='\r')
        results = run_c1(docs, model_name)
        _print_results(model_name, results)

    print(f"\n{sep}")
    print("C2 — 8 category classifiers combined (OR)")
    print(sep)
    for model_name in MODEL_NAMES:
        print(f"  Running {model_name}...", end='\r')
        results = run_c2(docs, model_name)
        _print_results(model_name, results)

    print()


if __name__ == '__main__':
    main()
