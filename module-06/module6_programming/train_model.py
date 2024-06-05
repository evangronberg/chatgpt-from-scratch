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
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = GPTModel(
        d_model=512, n_heads=16, layers=8,
        vocab_size=10000, max_seq_len=256
    )
    param_count = sum(p.numel() for p in model.parameters())
    print('Model has', param_count, 'parameters.')
    model = model.to(device)

    batch_size = 16
    microbatch_size = 4
    batch_count_loss_print = 100

    sequences = np.load('./sequences.npy', allow_pickle=True)
    sequences = torch.tensor(sequences)
    sequences_dataset = TensorDataset(sequences)
    sequences_loader = DataLoader(
        sequences_dataset, batch_size=microbatch_size)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    batch_count, batch_loss, batch_losses = 0, 0.0, []
    for microbatch_index, microbatch in tqdm(
        enumerate(sequences_loader), total=len(sequences_loader)
    ):
        prediction = model(microbatch[0].to(device))
        loss = loss_function(prediction, None) # TODO: Get `target`
        loss.backward()
        batch_loss += loss
        if microbatch_index % (batch_size / microbatch_size) == 0:
            optimizer.step()
            optimizer.zero_grad()
            batch_loss = batch_loss / batch_size
            batch_losses.append(batch_loss)
            if batch_count % batch_count_loss_print:
                print(f'BATCH {batch_count} LOSS: {batch_loss}')
                plt.plot(batch_losses)
                plt.savefig('./training_loss.png')
                plt.clf()
            batch_loss = 0.0
            batch_count += 1

    plt.plot(batch_losses)
    plt.savefig('./training_loss.png')
    torch.save(model.state_dict(), './model_weights.pt')

if __name__ == '__main__':
    train()
