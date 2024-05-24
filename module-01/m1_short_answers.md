# Module 1 Short Answers

Read the PyTorch with Examples tutorial to (re)familiarize yourself with PyTorch:

https://pytorch.org/tutorials/beginner/pytorch_with_examples.html

## Question 1

Explain the concept of a tensor, and the differences in PyTorch between a tensor, a parameter, and a gradient.

**Answer**

A tensor is, at its core, an n-dimensional matrix that can leverage graphical processing units (GPU), which enable neural network training and inference to be greatly sped up. The use of GPUs differentiates tensors from other similar data structures such as NumPy arrays. An additional differentiator is that tensors can store information on gradients and computational graphs, both of which are key for neural network computation (e.g., gradients are required for backpropagation).

A gradient is different from a generic tensor in that a gradient is a tensor that is _attached to_  a generic tensor, and it is calculated rather than directly instantiated. More specifically, it is the differential of the generic tensor with respect to another value (e.g., the loss resulting from a pass through a neural network).

Finally, a parameter is different from a generic tensor in that a parameter is a learnable value in a neural network. Parameters are defined by generic tensors, but their purpose and use is specific – they manipulate inputs to produce an output and are, over the course of time, modified by gradients (discussed above) to become better and better at transforming inputs into accurate outputs.

_Note: PyTorch offers a specific `Parameter` object, but the object itself is not discussed at any length in the tutorial provided, so it is not discussed here either._
