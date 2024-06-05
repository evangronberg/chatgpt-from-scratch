# External dependencies
import torch
import numpy as np
import matplotlib.pyplot as plt

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

    # Adjust as needed
    model = GPTModel(
        d_model=512, n_heads=16, layers=8,
        vocab_size=10000, max_seq_len=256
    )
    param_count = sum(p.numel() for p in model.parameters())
    print('Model has', param_count, 'parameters.')
    model = model.to(device)

    # TODO: Put new code here

    # Save model
    torch.save(model.state_dict(), './model_weights.pt')

if __name__ == '__main__':
    train()
