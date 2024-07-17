# External dependencies
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# Internal dependencies
from gpt import GPTModel

# Since we didn't really cover how to do this in lecture -
# this creates a learning rate schedule for you. Refer to the
# PyTorch docs for more info on using a scheduler.

# This one is designed for you to call scheduler.step() on every
# model update step.

def cosine_with_warmup_lr_scheduler(opt, total_steps, warmup_steps):
    def thunk(stepnum):
        if stepnum <= warmup_steps:
            # go from ~0 to 1.0
            prog = float(stepnum)/float(warmup_steps)
            lrmult = 0.00001 + prog
        else:
            # go from 1.0 to ~0
            steps_after_peak = stepnum-warmup_steps
            tail_steps = total_steps-warmup_steps
            prog = float(steps_after_peak) / float(tail_steps)
            lrmult = ((np.cos(3.141592*prog)+1.0)*0.5)*0.9 + 0.1
        return max(lrmult, 0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=thunk)
    return scheduler

"""
Complete the following method which trains a GPT model and saves a loss curve.
"""

def train() -> None:
    """
    Trains and saves a GPT model, also saves a loss curve.

    Arguments:
        None
    Return Values:
        None (saves model/loss curve)
    """
    # NOTE: Current settings take 6.5 hrs on M1 Pro
    d_model = 256
    n_heads = 8
    layers = 8
    vocab_size = 10000
    max_seq_len = 256
    batch_size = 32
    loss_check_cadence = 100

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = GPTModel(
        d_model=d_model, n_heads=n_heads, layers=layers,
        vocab_size=vocab_size, max_seq_len=max_seq_len
    )
    param_count = sum(p.numel() for p in model.parameters())
    print('Model has', param_count, 'parameters.')
    model = model.to(device)

    sequences = np.load('./sequences.npy', allow_pickle=True)
    sequences = torch.tensor(sequences)
    sequences_dataset = TensorDataset(sequences)
    sequences_loader = DataLoader(
        sequences_dataset, batch_size=batch_size)
    batch_count = len(sequences_loader)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = cosine_with_warmup_lr_scheduler(
        optimizer, total_steps=batch_count, warmup_steps=0.01*batch_count)

    loss_avgs, loss_sum = [], 0.0
    for batch_index, batch in tqdm(enumerate(sequences_loader), total=batch_count):
        optimizer.zero_grad()
        input = batch[0][:, :255].to(device)
        target = batch[0][:, 1:].to(device)
        prediction = model(input).transpose(1, 2)
        target = torch.nn.functional.one_hot(
            target, num_classes=vocab_size
        ).to(prediction.dtype).transpose(1, 2)
        loss = loss_function(prediction, target)
        loss_sum += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if (batch_index + 1) % loss_check_cadence == 0:
            loss_avg = float(loss_sum / loss_check_cadence)
            loss_avgs.append(loss_avg)
            loss_sum = 0.0
            print(f'AVERAGE LOSS FOR LAST {loss_check_cadence} BATCHES AT '
                  f'BATCH #{batch_index + 1}: {loss_avg}')
            plt.plot(loss_avgs)
            plt.xlabel(f'Token {loss_check_cadence}-Batch Counts '
                       f'(Batch Size of {batch_size})')
            plt.ylabel('Cross Entropy Loss')
            plt.title('Training Loss Curve')
            plt.savefig('./training_loss.png')
            plt.clf()

    plt.plot(loss_avgs)
    plt.xlabel(f'Token 100-Batch Counts (Batch Size of {batch_size})')
    plt.ylabel('Cross Entropy Loss')
    plt.title('Training Loss Curve')
    plt.savefig('./training_loss.png')
    torch.save(model.state_dict(), './model_weights.pt')

if __name__ == '__main__':
    train()
