from pathlib import Path
from typing import Iterable, Tuple

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[SOS]", "[POS]", "[EOS]"]


def get_or_build_tokenizer(
    cache_path: Path,
    sentences: Iterable[str],
    vocab_size: int,
    min_frequency: int = 2,
) -> Tokenizer:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return Tokenizer.from_file(str(cache_path))
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=min_frequency,
    )
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    tokenizer.save(str(cache_path))
    return tokenizer


def field_iter(dataset, field: str) -> Iterable[str]:
    for example in dataset:
        yield example[field]
