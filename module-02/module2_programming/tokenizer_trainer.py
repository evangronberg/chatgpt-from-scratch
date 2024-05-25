import json
from typing import Dict, List

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
    txt_file: str, vocabulary_size: int, base_vocabulary: List[str]
) -> None:
    """
    Trains a GPT-style tokenizer.

    Arguments:
        txt_file:        A string path to a text file of
                         data (i.e., `./data.txt`).
        vocabulary_size: Integer specifying the final vocab size.
        base_vocabulary: List of strings to add to the vocabulary
                         by default.
    Return Values:
        None (writes two files: `./vocab.txt` and `./merges.json`)
    """
    with open(txt_file, 'r') as file:
        corpus = file.readlines()[0]
        # corpus = file.read() # TODO: Re-implement for full corpus
    corpus_tokens = split_corpus_into_characters(corpus)

    merges = []
    vocabulary = initialize_vocabulary(base_vocabulary)
    while len(vocabulary) < vocabulary_size:
        corpus_token_counts = get_corpus_token_counts(corpus_tokens)

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
    prev_character_space = False
    for character_index, character in enumerate(corpus):
        if prev_character_space:
            prev_character_space = False
            continue
        if character != ' ':
            corpus_characters.append(character)
            prev_character_space = False
        else:
            corpus_characters.append(
                character + corpus[character_index + 1])
            prev_character_space = True
    return corpus_characters

def initialize_vocabulary(base_vocab: str) -> List[str]:
    """
    Creates a vocabulary using a single string comprised of
    all characters that the vocabulary should initially include.

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

def get_corpus_token_counts(corpus_tokens: List[str]) -> Dict[str, int]:
    """
    Gets the count of each token present in the given corpus.

    Arguments:
        corpus_tokens:       The tokens that comprise the corpus at hand.
    Return Values:
        corpus_token_counts: The count for each token.
    """
    corpus_token_counts = {}
    current_word_tokens = []
    for token in corpus_tokens:
        if token[0] == ' ':
            word_token_counts = get_word_token_counts(current_word_tokens)
            corpus_token_counts = {
                token: corpus_token_counts.get(token, 0) +\
                    word_token_counts.get(token, 0)
                for token in set(corpus_token_counts) | set(word_token_counts)
            }
            current_word_tokens = []
            continue
        current_word_tokens.append(token)
    return corpus_token_counts

def get_word_token_counts(word_tokens: List[str]) -> Dict[str, int]:
    """
    Gets the count of each token present in the given word.

    Arguments:
        word_tokens:       The tokens that comprise the word at hand.
    Return Values:
        word_token_counts: The count for each token.
    """
    word_token_counts = {}
    for token_index in range(len(word_tokens) - 1):
        token_pair = word_tokens[token_index] + word_tokens[token_index + 1]
        if token_pair not in word_token_counts.keys():
            word_token_counts[token_pair] = 1
        else:
            word_token_counts[token_pair] += 1
    return word_token_counts

if __name__ == '__main__':

    base = 'abcdefghijklmnopqrstuvwxyz'
    base += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    base += '0123456789'
    base += '!@#$%^&*()_+-=[]{}|;":,.<>/?`~ '
    base += '\\'
    base += "'"

    train_tokenizer('./data.txt', len(base) + 1000, [c for c in base])
