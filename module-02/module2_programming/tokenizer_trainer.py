import json
from typing import List

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
        base_vocabulary: List of strings to add to the vocabulary by default.
    Return Values:
        None (writes two files: `./vocab.txt` and `./merges.json`)
    """
    # TODO

if __name__ == '__main__':

    base = 'abcdefghijklmnopqrstuvwxyz'
    base += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    base += '0123456789'
    base += '!@#$%^&*()_+-=[]{}|;":,.<>/?`~ '
    base += '\\'
    base += "'"

    train_tokenizer('./data.txt', len(base) + 1000, [c for c in base])
