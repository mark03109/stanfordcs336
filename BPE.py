from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
import tiktoken

class Tokenizer(ABC):
    """Abstract interface for a tokenizer"""
    def encode(self, string: str) -> list[int]:
        raise NotImplementedError

    def decode(self, indices: list[int]) -> str:
        raise NotImplementedError


class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points"""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))    

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))


#str --encode("utf-8")--> bytes
#bytes --decode("utf-8")--> str
class ByteTokenizer(Tokenizer):
    """Represent a string as a sequence of bytes"""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")
        return list(map(int, string_bytes))

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)
        return string_bytes.decode("utf-8")


def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    """Return 'indices', but with all instances of 'pair' replaced with 'new_index'"""
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i+1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
        return new_indices


@dataclass(frozen = True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer"""
    vocab: dict[int, bytes] # index -> bytes
    merge: dict[tuple[int,int], int] # index1, index2 -> new_index


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary"""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))
        # Notes: this is a very slow implementation
        for pair, new_index in self.params.merge.items():
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))
        string = b"".join(bytes_list).decode("utf-8")# b"" means empty bytes
        return string


def get_compression_ratio(string: str, indices: list[int]) -> float:
    """Given 'string' that has been tokenlized into 'indices', return the number of UTF-8 bytes per token"""
    num_bytes = len(bytes(string, encoding = "utf-8"))
    num_tokens = len(indices)
    return num_bytes / num_tokens

def get_gpt5_tokenizer():
    return tiktoken.get_encoding("o200k_base")

def tokenization_examples():
    tokenizer = get_gpt5_tokenizer()
    string = "Hello, 🌍! 你好!"
    indices = tokenizer.encode(string)
    reconstructed_string = tokenizer.decode(indices)
    assert string == reconstructed_string

    compression_ratio = get_compression_ratio(string, indices)

    vocabulary_size = tokenizer.n_vocab

    """user test"""
    print(f"vocalbulary_size: {vocabulary_size}\nindices: {indices}\n")
    new_indices = []
    for i in indices:
        new_indices.append(i)
        print(f"{new_indices} {tokenizer.decode(new_indices)}\n")

if __name__ == "__main__":
    tokenization_examples()
