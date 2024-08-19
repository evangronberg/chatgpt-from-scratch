# Python-native dependencies
import json
from collections import defaultdict
from typing import Dict, List, Tuple

# External dependencies
from tqdm import tqdm

def train_tokenizer(
    txt_file: str, vocab_size: int 
) -> None:
    """
    """
    vocabulary = [
        bytes([byte]) for byte in range(256)
    ]
    with open(txt_file, 'r') as corpus_file:
        corpus = corpus_file.read()
    corpus_encoded = corpus.encode('utf-8')
    corpus_tokens = [
        bytes([byte]) for byte in list(corpus_encoded)
    ]
    merges = []
    initial_vocabulary_size = len(vocabulary)
    for _ in tqdm(range(vocab_size - initial_vocabulary_size)):
        corpus_token_pair_counts =\
            get_corpus_token_pair_counts(corpus_tokens)
        most_common_token_pair = max(
            corpus_token_pair_counts,
            key=corpus_token_pair_counts.get
        )
        merges.append(most_common_token_pair)
        vocabulary.append(
            most_common_token_pair[0] +\
            most_common_token_pair[1]
        )
        corpus_tokens = update_corpus_tokens_with_token_pair(
            corpus_tokens, most_common_token_pair
        )
    with open(
        txt_file.replace('.txt', '_vocab.txt'), 'w'
    ) as vocabulary_file:
        for token in vocabulary:
            vocabulary_file.write(f'{token}\n')
    with open(
        txt_file.replace('.txt', '_merges.json'), 'w'
    ) as merges_file:
        json.dump(merges, merges_file)

def get_corpus_token_pair_counts(
    corpus_tokens: List[bytes]
) -> Dict[Tuple[bytes, bytes], int]:
    """
    """
    corpus_token_pair_counts = defaultdict(int)
    for token_index in range(len(corpus_tokens) - 1):
        token_pair = (
            corpus_tokens[token_index],
            corpus_tokens[token_index + 1]
        )
        corpus_token_pair_counts[token_pair] += 1
    return corpus_token_pair_counts
        
def update_corpus_tokens_with_token_pair(
    corpus_tokens: List[bytes], token_pair: Tuple[bytes, bytes]
) -> List[bytes]:
    """
    """
    updated_corpus_tokens = []
    previous_token_was_in_pair = False
    for token_index in range(len(corpus_tokens) - 1):
        if previous_token_was_in_pair:
            previous_token_was_in_pair = False
            continue
        if corpus_tokens[token_index] == token_pair[0] and \
            corpus_tokens[token_index + 1] == token_pair[1]:
            updated_corpus_tokens.append(
                token_pair[0] + token_pair[1]
            )
            previous_token_was_in_pair = True
        else:
            updated_corpus_tokens.append(
                corpus_tokens[token_index]
            )
            previous_token_was_in_pair = False
    return updated_corpus_tokens

if __name__ == '__main__':
    train_tokenizer(
        txt_file='./test_txt_files/test1.txt',
        vocab_size=100
    )
