from abc import ABC
from dataclasses import dataclass
from collections import defaultdict
import tiktoken
import os

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

def output_tokenizer(tokenizer, path: str):
    """Write out the vocabulary of tokenizer to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not os.path.exists(path):
        vocab = [
            b.decode("utf-8", errors="replace")
            for b in tokenizer.token_byte_values()
        ]

        with open(path, "w", encoding="utf-8") as f:
            for token in vocab:
                f.write(token + "\n")

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

    output_tokenizer(tokenizer, "var/gpt_tokenizer_vocab.txt")


def count_adjacent_pairs(indices: list[int]) -> dict[tuple[int, int], int]:
    """Return a dictionary mapping each adjacent pair of tokens in 'indices' to the number of times it occurs"""
    counts = defaultdict(int)# create a dict with default value
    for index1, index2 in zip(indices, indices[1:]):
        counts[(index1, index2)] += 1
    return counts 

def train_bpe(string: str, num_merges: int) ->BPETokenizerParams:
    indices = list(map(int, string.encode("utf-8")))
    merges: dict[tuple[int, int], int] = {}
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)} # index ->bytes
    # if the arg of bytes([x]) is a list, then it convert [x] to a bytes obeject liks b"abc"
    # but bytes(x) means create a  bytes with x byte"\x00"

    for i in range(num_merges):
        # count the number of occurences of each pair of tokens 
        counts = count_adjacent_pairs(indices)

        # find the most commen pair, the first one if many with the same max value
        pair = max(counts, key=counts.get)

        # Merge that pair
        new_index = 256 + i
        merges[pair] = new_index
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
        indices = merge(indices, pair, new_index)

    compression_ratio = get_compression_ratio(string, indices)

    return BPETokenizerParams(vocab, merges)
        


def bpe_tokenizer():
    string = "the cat in the hat"
    params = train_bpe(string, num_merges=4)
    tokenizer = BPETokenizer(params)
    string = "the quick brown fat fox"
    tokens = tokenizer.encode(string)
    print(tokens)
    reconstructed_string = tokenizer.decode(tokens)
    assert string == reconstructed_string
    print(reconstructed_string)


if __name__ == "__main__":
    bpe_tokenizer()



