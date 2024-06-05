# Python-native dependencies
import math

# External dependencies
import numpy as np
from tqdm import tqdm

# Internal dependencies
from hftokenizer import HFTokenizer

def construct_dataset(
    data_txt_file: str, sequence_length: int = 256
) -> np.ndarray:
    """
    Uses the trained tokenizer to convert samples to token IDs then pack them
    into a training set represented as a NumPy array of size (n_sequences,
    sequence_length). This array is shuffled then saved to the file system
    using NumPy's np.save() method.

    Arguments:
        data_txt_file:   A string path to a text file containing training
                         data, one sample per line.
        sequence_length: The desired length of each training sequence.
    Return Values:
        sequences:       A NumPy array of size (sequences, sequence_length).
                         (Also saves this array to the file system.)
    """
    tokenizer = HFTokenizer()
    tokenizer.load()

    with open(data_txt_file, 'r') as f:
        samples = f.readlines()

    tokenized_samples = [
        tokenizer.encode(sample) for sample in tqdm(samples)
    ]
    sequences = []
    next_sequence = []
    for sample in tokenized_samples:
        if len(sample) + len(next_sequence) <= sequence_length:
            next_sequence += sample
        else:
            sample_slice = sequence_length - len(next_sequence)
            next_sequence += sample[:sample_slice]
            sequences.append(next_sequence)
            next_sequence = sample[sample_slice:]
            if len(next_sequence) > sequence_length:
                n_new_sequences = math.ceil(
                    len(next_sequence) / sequence_length)
                for new_sequence_index in range(n_new_sequences - 1):
                    new_sequence = next_sequence[
                        new_sequence_index * sequence_length:
                        (new_sequence_index + 1) * sequence_length
                    ]
                    sequences.append(new_sequence)
                next_sequence = next_sequence[
                    (new_sequence_index + 1) * sequence_length:
                ] # NOTE: If we're on the last sample, this does sequence
                  #       does not get added to `sequences` since it is
                  #       not a complete, 256-length sequence
    sequences = np.array(sequences)
    sequences = np.random.shuffle(sequences)
    np.save('./sequences.npy', sequences)
    return sequences

if __name__ == '__main__':
    construct_dataset('./data.txt', 256)
