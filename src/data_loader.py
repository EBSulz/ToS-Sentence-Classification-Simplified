from pathlib import Path

CATEGORIES = ['a', 'ch', 'cr', 'j', 'law', 'ltd', 'ter', 'use']

_CATEGORY_FOLDER = {
    'a': 'Labels_A',
    'ch': 'Labels_CH',
    'cr': 'Labels_CR',
    'j': 'Labels_J',
    'law': 'Labels_LAW',
    'ltd': 'Labels_LTD',
    'ter': 'Labels_TER',
    'use': 'Labels_USE',
}


def _read_sentences(path: Path) -> list[str]:
    with open(path, encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f if line.strip()]


def _read_labels(path: Path) -> list[int]:
    with open(path, encoding='utf-8') as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_corpus(data_dir: Path) -> list[dict]:
    """
    Load all ToS documents.

    Returns a list of dicts:
        name          : str              - document name (stem of filename)
        sentences     : list[str]        - raw tokenised sentences
        labels        : list[int]        - general labels (1 unfair, -1 fair)
        cat_labels    : dict[str, list[int]] - per-category labels
    """
    sentences_dir = data_dir / 'Sentences'
    labels_dir = data_dir / 'Labels'

    docs = []
    for txt_file in sorted(sentences_dir.glob('*.txt')):
        name = txt_file.stem
        sentences = _read_sentences(txt_file)
        general_labels = _read_labels(labels_dir / txt_file.name)

        n = min(len(sentences), len(general_labels))
        sentences = sentences[:n]
        general_labels = general_labels[:n]

        cat_labels: dict[str, list[int]] = {}
        for cat, folder in _CATEGORY_FOLDER.items():
            cat_file = data_dir / folder / txt_file.name
            if cat_file.exists():
                raw = _read_labels(cat_file)
                cat_labels[cat] = raw[:n]
            else:
                cat_labels[cat] = [-1] * n

        docs.append({
            'name': name,
            'sentences': sentences,
            'labels': general_labels,
            'cat_labels': cat_labels,
        })

    return docs
