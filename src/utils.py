import json
import re
from tqdm import tqdm
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

class Config():

    #Model architecture
    vocab_size = 512
    n_seq = 512
    d_embd = 256
    n_head = 8
    n_head_q = 8
    n_kv_head = 4
    n_layer = 8
    dropout = 0.2
    
    #Training
    batch_size = 160
    learning_rate = 3e-4
    max_epochs = 10
    weight_decay = 0.01

    #Generation
    max_new_tokens = 512


class CharLevelTokenizer:
    def __init__(self):
        self.char_to_idx = {}  # Character to index mapping
        self.idx_to_char = {}  # Index to character mapping
        self.vocab_size = 0

    def fit(self, texts: List[str]) -> None:
        """Build vocabulary from list of texts.

        Args:
            texts (List[str]): A list of strings from which to build the vocabulary.
        """
        # Get unique characters from all texts
        unique_chars = sorted(set(''.join(texts)))

        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(unique_chars)}
        self.vocab_size = len(unique_chars)

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token indices.

        Args:
            text (str): The input text to encode.

        Returns:
            List[int]: A list of token indices corresponding to the input text.
        """
        return [self.char_to_idx[char] for char in text]

    def decode(self, indices: List[int]) -> str:
        """Convert list of token indices back to text.

        Args:
            indices (List[int]): A list of token indices to decode.

        Returns:
            str: The decoded text.
        """
        return ''.join(self.idx_to_char[idx] for idx in indices)

    def get_vocab(self) -> Dict[str, Any]:
        """Return the vocabulary dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing character to index mapping, index to character mapping, and vocabulary size.
        """
        return {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size
        }

    def save_vocab(self, filename: str) -> None:
        """Saves the vocab as <filename>.json.

        Args:
            filename (str): The name of the file to save the vocabulary.
        """

        vocab = self.get_vocab()

        with open(f"{filename}.json", 'w', encoding='utf-8') as f:
            json.dump(vocab,f, indent = 2)


    def set_vocab(self, vocab: Dict[str, Any]) -> None:
        """Sets up the tokenizer with given Vocabulary.

        Args:
            vocab (Dict[str, Any]): A dictionary containing character to index mapping, index to character mapping, and vocabulary size.
        """

        self.char_to_idx = vocab['char_to_idx']
        self.idx_to_char = vocab['idx_to_char']
        self.vocab_size = vocab['vocab_size']

class BPETokenizer:
    def __init__(self, vocab_size: int = 512):
        """
        Initializes the BPE Tokenizer with a specified vocabulary size.

        Args:
            vocab_size (int): The maximum size of the vocabulary. Default is 512.
        """
        self.vocab_size = vocab_size
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}
        self.pattern = None

    def _get_stats(self, ids: List[int]) -> Dict[tuple, int]:
        """
        Count frequency of adjacent pairs in the token IDs.

        Args:
            ids (List[int]): List of token IDs.

        Returns:
            Dict[tuple, int]: A dictionary with pairs of token IDs as keys and their counts as values.
        """
        counts = {}
        i = 0
        while i < len(ids) - 1:
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
            i += 1
        return counts

    def _merge(self, ids: List[int], pair: tuple, idx: int) -> List[int]:
        """
        Merge the most frequent pair of token IDs.

        Args:
            ids (List[int]): List of token IDs.
            pair (tuple): The pair of token IDs to merge.
            idx (int): The index for the new merged token.

        Returns:
            List[int]: Updated list of token IDs after merging.
        """
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def fit(self, texts: List[str], verbose: bool = False) -> None:
        """
        Fit the BPE tokenizer on the provided texts.

        Args:
            texts (List[str]): List of texts to fit the tokenizer on.
            verbose (bool): If True, prints the initial and final vocabulary sizes. Default is False.

        Returns:
            None
        """
        unique_chars = sorted(set(''.join(texts)))
        self.idx_to_token = {idx: char for idx, char in enumerate(unique_chars)}
        self.token_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
        current_vocab_size = len(unique_chars)

        if verbose:
            print(f"Initial vocabulary size: {current_vocab_size}")

        ids = [self.token_to_idx[char] for char in ''.join(texts)]
        num_merges = self.vocab_size - current_vocab_size

        for i in tqdm(range(num_merges), desc="Training BPE"):
            stats = self._get_stats(ids)
            if not stats:
                break

            pair = max(stats.items(), key=lambda x: x[1])[0]
            ids = self._merge(ids, pair, current_vocab_size)

            merged_token = self.idx_to_token[pair[0]] + self.idx_to_token[pair[1]]
            self.idx_to_token[current_vocab_size] = merged_token
            current_vocab_size += 1

        self.token_to_idx = {token: idx for idx, token in self.idx_to_token.items()}

        if verbose:
            print(f"Final vocabulary size: {current_vocab_size}")

        tokens_by_length = sorted(self.token_to_idx.keys(), key=len, reverse=True)
        self.pattern = re.compile("|".join(map(re.escape, tokens_by_length)))

    def encode(self, text: str) -> List[int]:
        """
        Encode a given text into a list of token IDs.

        Args:
            text (str): The input text to encode.

        Returns:
            List[int]: A list of token IDs corresponding to the input text.
        """
        if self.pattern is None:
            raise ValueError("Tokenizer must be fitted before encoding")

        tokens = self.pattern.findall(text)
        return [self.token_to_idx[token] for token in tokens]

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back into text.

        Args:
            ids (List[int]): List of token IDs to decode.

        Returns:
            str: The decoded text.
        """
        return ''.join(self.idx_to_token[idx] for idx in ids)

    def _get_vocab(self) -> Dict[str, Any]:
        """
        Get the vocabulary of the tokenizer.

        Returns:
            Dict[str, Any]: A dictionary containing token to index mapping, index to token mapping, and vocabulary size.
        """
        return {
            'token_to_idx': self.token_to_idx,
            'idx_to_token': self.idx_to_token,
            'vocab_size': self.vocab_size
        }

    def save_vocab(self, filename: str) -> None:
        """
        Save the tokenizer vocabulary to a JSON file.

        Args:
            filename (str): The name of the file to save the vocabulary.

        Returns:
            None
        """
        vocab = self._get_vocab()
        with open(f"{filename}.json", 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

    def load_vocab(self, filename: str) -> None:
        """
        Load the tokenizer vocabulary from a JSON file.

        Args:
            filename (str): The name of the file to load the vocabulary from.

        Returns:
            None
        """
        with open(f"{filename}.json", 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self.token_to_idx = vocab['token_to_idx']
        self.idx_to_token = vocab['idx_to_token']
        self.vocab_size = vocab['vocab_size']

        tokens_by_length = sorted(self.token_to_idx.keys(), key=len, reverse=True)
        self.pattern = re.compile("|".join(map(re.escape, tokens_by_length)))


def load_dataset(filepath: str) -> str:
    with open(filepath, 'r') as f:
        return f.read()


class LyricsDataset(Dataset):
    def __init__(self, lyrics, block_size, tokenizer):
        self.tok_lyrics = tokenizer.encode(lyrics)
        self.block_size = block_size

    def __len__(self):
        return len(self.tok_lyrics) - self.block_size

    def __getitem__(self, idx):

        dix = self.tok_lyrics[idx : idx + self.block_size + 1]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
    
