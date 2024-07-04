# Module 5 Short Answers

Read the article "Transformers from Scratch" by Peter Bloem. Note that while this provides a great explanation of transformers, some of the specifics are slightly dated. If you see an implementation discrepancy between this article and our class lecture notes, please defer to the lecture notes.

https://peterbloem.nl/blog/transformers

## Question 1

What are the main differences between "modern" transformers and the original encoder/decoder transformer ("historical baggage" as Bloem calls it)? Why has the model architecture changed?

**Answer**

The glaring difference between "modern" transformers and the original encoder/decoder transformers is the typical lack of an encoder component in most transformer implementations. The encoder component gave transformers a more symmetric architecture, akin to an autoencoder or perhaps a generative adversarial network (GAN). Per Bloem, this change was a simplification due to the sufficiency of a decoder-only architecture.

Specific modern transformers introduce further differences from the original transformer:

- BERT implemented bidirectional encoding and generic pre-training (i.e., training a generic model that is then tuned for specific applications, which includes adding a new task-specific layer).
- Transformer-XL divided input into segments (for the sake of handling larger inputs) and computed self-attention over both the current and previous segments.
- Sparse transformers did not compute self-attention over the entire input, instead only computing self-attention for pre-selected (i.e., not learned) token pairings. This resulted in attention matrices of only size $n\sqrt{n}$ rather than $n^2$.
