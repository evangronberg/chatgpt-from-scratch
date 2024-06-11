# Module 7 Short Answers

Read a paper that was not discussed in class lecture: the Longformer paper. This paper is one of many that have attempted to make a more efficient attention mechanism.

Read sections 1 through 4 of the paper:

https://arxiv.org/pdf/2004.05150

## Question 1

Why does the memory complexity of a transformer expand quadratically when the input sequence only expands linearly? How does this limit our ability to build larger and larger models?

**Answer**

The memory requirements of the standard self-attention mechanism scale quadratically with input sequence length since the standard mechanism compares every token in the sequence with every token in the sequence (inclusive of itself!). The result of these comparisons can be thought of as "connection values." Suppose we denote the number of tokens in the input sequence as $n$; for a given token, then, $n$ "connection values" must be computed, and because there are $n$ tokens, our attention matrix is of size $n^2$. Whether dealing with time or memory complexity, an algorithm that scales quadratically like this one is considered unideal; for self-attention in particular, where $n$ needs to scale to the thousands for maximum model quality, this unideal complexity is especially pronounced.

## Question 2

How does the Longformer try to improve on this complexity?

**Answer**

In essence, the Longformer addresses this problem by "sparsifying" the $n^2$ self-attention matrix. Quite notably, the "sparsification" approach scales _linearly_ (i.e., $n$) rather than quadratically. This is exceptional – oftentimes, improvements to quadratic algorithms only scale performance down to $n \log (n)$ or some power of $n$ less than 2 but greater than 1. The Longformer paper actually offers multiple variations of this "sparsification" method, but they are all rooted in the same concept: "sliding window attention." The sliding window method takes advantage of an intuitive characteristic of language, that _local_ context is paramount. The method thus "employs a fixed-size window surrounding each token." That is, instead of having to compare every token in a sequence with every token in the sequence, a given token is only compared with a fixed number of surrounding tokens. The variations of this technique incorporate "dilation" (i.e., introducing gaps in the window surrouding each token), as well as the re-incorporation of global context for select tokens in the sequence.
