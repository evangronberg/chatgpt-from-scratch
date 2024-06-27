"""
Module implementing rotary position embeddings (RoPE).
"""

# External depedencies
import torch

def apply_rotary_position_embeddings(tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary position embeddings to the given tensor.

    Arguments:
        TBA
    Return Values:
        TBA
    """
    seq_len, d_model = tensor.shape[1], tensor.shape[2]

    # We establish one theta value per pair of values in the embeddings.
    # The sequence of theta values is initially (0, 2, 4, ..., d_model).
    # We divide by d_model to normalize the values within [0, 1].
    thetas = torch.arange(0, d_model, 2).float() / d_model
    print(thetas.shape)
    print(thetas.data)

    # We need to get each position value for multiplying the thetas by
    positions = torch.arange(0, seq_len).float()
    print(positions.shape)
    print(positions.data)

    theta = torch.exp(-torch.arange(0, seq_len).float()[:, None] * theta * 2 * torch.pi)
    print(theta.data)

    angles = thetas * positions

    cos_theta = torch.cos(angles)
    sin_theta = torch.sin(angles)

    tensor_cos = tensor * cos_theta[:, :, None]
    tensor_sin = tensor * sin_theta[:, :, None]

    return torch.cat([tensor_cos, tensor_sin], dim=-1)

if __name__ == '__main__':
    seq_len = 6
    batch_size = 32
    d_model = 128

    x = torch.rand(batch_size, seq_len, d_model)
    x_rotary = apply_rotary_position_embeddings(x)
    print(x.shape, x_rotary.shape)
