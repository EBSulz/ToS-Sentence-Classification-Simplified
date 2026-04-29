import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

_STOP_WORDS: set[str] = set(stopwords.words('english'))

# Penn Treebank bracket tokens produced by Stanford CoreNLP
_PTB_TOKENS: set[str] = {'-lrb-', '-rrb-', '-lsb-', '-rsb-', '-lcb-', '-rcb-'}

# Matches tokens that contain no alphanumeric characters
_NON_ALPHA = re.compile(r'^[^a-z0-9]+$')


def clean(sentence: str) -> str:
    """
    Normalise a pre-tokenised, lowercased sentence:
      - drop PTB bracket tokens
      - drop punctuation-only tokens
      - drop stopwords
    Returns a space-joined string ready for TF-IDF.
    """
    tokens = sentence.lower().split()
    out: list[str] = []
    for token in tokens:
        if token in _PTB_TOKENS:
            continue
        if _NON_ALPHA.match(token):
            continue
        if token in _STOP_WORDS:
            continue
        out.append(token)
    return ' '.join(out)
