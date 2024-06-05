# External dependencies
import torch
import numpy as np
from tqdm import tqdm

# Internal dependencies
from hftokenizer import HFTokenizer

def construct_dataset(data_txt_file: str, sequence_length: int = 256) -> None:
    """
    This method should use the trained tokenizer to convert samples to
    token_ids, and then pack them into a training set represented as a
    2D array of size (sequences, sequence_length).

    You can save this training set in whatever format you wish for loading
    into the training script. I recommend using numpy's np.save() method
    or the pickle module.

    The saved data should be shuffled so we can directly load it and train
    on it in the training script.

    Arguments:
        data_txt_file:   A string path to a text file containing training
                         data, one sample per line.
        sequence_length: The desired length of each training sequence.
    Return Values:
        TBA
    """

    # Construct tokenizer
    tokenizer = HFTokenizer()
    tokenizer.load()

    # Get all samples
    f = open(data_txt_file, "r")
    samples = f.readlines()

    # TODO: Put new code here

if __name__ == '__main__':
    construct_dataset('./data.txt', 256)
