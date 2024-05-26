# Python-native dependencies
import json
from typing import List

"""
This class should be constructed with trained tokenizer data:
vocab_file : a string path to a vocab.txt file
merges_file : a string path to a merges.json file

The class should implement two methods:
encode(string): returns a list of integer ids (tokenized text)
decode(list_of_ids): returns a string re-assembled from token ids

A good sanity check is that decode(encode(x)) should return x.

You may assume that only a single sample is passed in at a time (no batching).
You can add additional methods, classes, etc as you find helpful.
"""

class Tokenizer:
    """
    Leverages existing vocabulary and merging files to
    tokenize (encode) and de-tokenize (decode) strings.

    Arguments:
        vocab_file:  A .txt file listing vocabulary tokens in order.
        merges_file: A .json file listing tokenizer merges in order.
    """
    def __init__(self, vocab_file: str, merges_file: str) -> None:
        with open(vocab_file, 'r') as vocab_file_stream:
            self.vocab = [
                token.replace('\n', '')
                for token in vocab_file_stream.readlines()
            ]
        with open(merges_file, 'r') as merges_file_stream:
            self.merges = json.load(merges_file_stream)

    def encode(self, string: str) -> List[int]:
        """
        Tokenizes a string into a series of integers (i.e., token IDs).

        Arguments:
            string:         The string to be encoded.
        Return Values:
            encoded_string: The string represented as a list of integers.
        """
        words = []
        current_word_characters = []
        for character in string:
            if character == ' ':
                words.append(current_word_characters)
                current_word_characters = [character]
            else:
                current_word_characters.append(character)
        if current_word_characters:
            words.append(current_word_characters)

        for word_index, word in enumerate(words):
            for merge in self.merges:
                if merge[0] + merge[1] in ''.join(word):
                    # We do this to ensure that, if there are multiple instances
                    # of this merge in a given word, all instances are handled
                    replacement_indices = [
                        index for index in range(len(word) - 1)
                        if word[index] + word[index + 1] == merge[0] + merge[1]
                    ]
                    for index in replacement_indices:
                        word[index] = merge[0] + merge[1]
                        del word[index + 1]
                    words[word_index] = word
        tokens = [token for word in words for token in word]

        encoded_string = [self.vocab.index(token) for token in tokens]
        return encoded_string

    def decode(self, list_of_integers: List[int]) -> str:
        """
        De-tokenizes a list of integers into a string.

        Arguments:
            list_of_integers: A list of token IDs.
        Return Values:
            string:           The decoded string.
        """
        string = ''
        for integer in list_of_integers:
            string += self.vocab[integer]
        return string

if __name__ == '__main__':

    tok = Tokenizer('./vocab.txt', './merges.json')
    x = tok.encode('Peter piper picked a peck of pickled peppers.')
    print(x)
    x = tok.decode(x)
    print(x)
