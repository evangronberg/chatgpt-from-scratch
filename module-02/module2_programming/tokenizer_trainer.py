# Python-native dependencies
import json
from collections import defaultdict
from typing import Dict, List, Tuple

# External dependencies
from tqdm import tqdm

"""
Your assignment is to implement BPE in the following method. You can add
classes or other routines if you find them helpful. 

This method should save two output files:
./vocab.txt : a list of the final vocabulary in order, one entry per line
./merges.json : a list of tuples of merges, in order

NOTE: Typically these file extensions are reversed (the vocabulary is a
json file and the merge list is a txt file), but for our purposes this way
seems simplier.

Does not need to return anything.

-------

This should implement a GPT-style tokenizer which prefixes words with a space.
You can assume that the base vocabulary contains all single characters
that will occur.
You do NOT need to worry about using a placeholder token in place of a space. 
You do NOT need to worry about special tokens (pad, bos, eos, unk, etc.). We
have not covered these yet.
"""

def train_tokenizer(
    txt_file: str, vocab_size: int, base_vocabulary: List[str]
) -> None:
    """
    Trains a GPT-style tokenizer.

    Arguments:
        txt_file:        A string path to a text file of
                         data (i.e., `./data.txt`).
        vocab_size:      Integer specifying the final vocab size.
        base_vocabulary: List of strings to add to the vocabulary
                         by default.
    Return Values:
        None (writes two files: `./vocab.txt` and `./merges.json`)
    """
    with open(txt_file, 'r') as corpus_file:
        corpus = corpus_file.read()
    corpus_tokens = split_corpus_into_characters(corpus)

    merges = []
    vocabulary = initialize_vocabulary(base_vocabulary)
    initialize_vocabulary_size = len(vocabulary)
    for _ in tqdm(range(vocab_size - initialize_vocabulary_size)):
        corpus_pair_counts = get_corpus_pair_counts(corpus_tokens)
        most_common_pair = max(
            corpus_pair_counts, key=corpus_pair_counts.get)
        merges.append(most_common_pair)
        vocabulary.append(''.join(most_common_pair))
        corpus_tokens = update_corpus_tokens_with_pair(
            corpus_tokens, most_common_pair)

    with open('vocab.txt', 'w') as vocab_file:
        for token in vocabulary:
            vocab_file.write(f'{token}\n')
    with open('merges.json', 'w') as merges_file:
        json.dump(merges, merges_file)

def split_corpus_into_characters(corpus: str) -> List[str]:
    """
    Splits a single-string corpus into a list of individual
    characters in a GPT style (i.e., prefixes words with a space).

    Arguments:
        corpus:            A corpus of text represented
                           as a single string.
    Return Values:
        corpus_characters: A corpus of text split up into
                           individual characters.
    """
    corpus_characters = []
    prev_character_was_space = False
    for character_index, character in enumerate(corpus):
        if prev_character_was_space:
            prev_character_was_space = False
            continue
        if character != ' ':
            corpus_characters.append(character)
            prev_character_was_space = False
        else:
            corpus_characters.append(
                character + corpus[character_index + 1])
            prev_character_was_space = True
    return corpus_characters

def initialize_vocabulary(base_vocab: str) -> List[str]:
    """
    Creates a vocabulary using a single string comprised of
    all characters that the vocabulary should initially include.

    NOTE: This vocabulary accounts for spaces GPT-style (i.e., each
          letter has a version with a space appended to it).

    Arguments:
        base_vocab: A string of all characters the vocabulary
                    should contain.
    Return Values:
        vocab:      A list of all characters as an initial
                    vocabulary.
    """
    vocab = []
    for character in base_vocab:
        if character == ' ':
            continue
        elif character.isalpha():
            vocab.append(character)
            vocab.append(' ' + character)
        else:
            vocab.append(character)
    return vocab

def get_corpus_pair_counts(
    corpus_tokens: List[str]
) -> Dict[Tuple[str, str], int]:
    """
    Gets the count of each token present in the given corpus.

    Arguments:
        corpus_tokens:      The tokens that comprise the corpus at hand.
    Return Values:
        corpus_pair_counts: The count for each token.
    """
    corpus_pair_counts = defaultdict(int)
    current_word_tokens = []
    for token in corpus_tokens:
        # If the token starts with a space, then we're starting a new word
        if token[0] == ' ':
            corpus_pair_counts = update_corpus_pair_counts_with_word(
                corpus_pair_counts, current_word_tokens
            )
            current_word_tokens = [token]
        else:
            current_word_tokens.append(token)
    return dict(corpus_pair_counts)

def update_corpus_pair_counts_with_word(
    corpus_pair_counts: Dict[Tuple[str, str], int], word_tokens: List[str]
) -> Dict[Tuple[str, str], int]:
    """
    Updates the pair counts for the corpus with
    the tokens comprising the next word.

    Arguments:
        corpus_pair_counts: The current count for each token.
        word_tokens:        The tokens in the word to use for the update.
    Return Values:
        corpus_pair_counts: The updated count for each token.
    """
    for token_index in range(len(word_tokens) - 1):
        token_pair = (word_tokens[token_index], word_tokens[token_index + 1])
        corpus_pair_counts[token_pair] += 1
    return corpus_pair_counts

def update_corpus_tokens_with_pair(
    corpus_tokens: List[str], pair: Tuple[str, str]
) -> List[str]:
    """
    Produces a new corpus with the given new pair reflected.

    Arguments:
        corpus_tokens:         The tokens in the old corpus.
        pair:                  The new token pair to update
                               the corpus with.
    Return Values:
        updated_corpus_tokens: The tokens for the new corpus.
    """
    prev_token_was_pair = False
    updated_corpus_tokens = []
    for token_index in range(len(corpus_tokens)):
        if prev_token_was_pair:
            prev_token_was_pair = False
            continue
        if corpus_tokens[token_index] == pair[0] and \
            corpus_tokens[token_index + 1] == pair[1] and \
            corpus_tokens[token_index + 1][0] != ' ':
            updated_corpus_tokens.append(''.join(pair))
            prev_token_was_pair = True
        else:
            updated_corpus_tokens.append(corpus_tokens[token_index])
            prev_token_was_pair = False
    return updated_corpus_tokens

if __name__ == '__main__':

    base = 'abcdefghijklmnopqrstuvwxyz'
    base += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    base += '0123456789'
    base += '!@#$%^&*()_+-=[]{}|;":,.<>/?`~ '
    base += '\\'
    base += "'"

    train_tokenizer('./data.txt', len(base) + 1000, [c for c in base])
