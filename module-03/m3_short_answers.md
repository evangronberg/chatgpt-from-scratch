# Module 3 Short Answers

Watch the Computerphile video on Word2Vec:

https://www.youtube.com/watch?v=gQddtTdmG_8

Also read the following article on sinusoidal positional encodings:

https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/

## Question 1

In the readings, we see an old version of a positional encoding (sine function) and an even older version of word embeddings (Word2Vec). What are the pros and cons of learning these embeddings directly as opposed to a pre-defined/pre-learned embedding as described in the readings?

**Answer**

The greatest advantage of learning embeddings directly is that they become incredibly well-tuned to the dataset and use case at hand. This leads to increased model quality, although it should be noted that in the case of positional embeddings, this module's lecture slides mention that both sinusoidal embeddings and learned embeddings "[i]n practice...work equally well." Thus, this advantage applies primarily to word embeddings – and indeed, learning word embeddings directly is now standard practice for today's language models.

The greatest disadvantage of learning embeddings directly is the computational overhead introduced, particularly for word embeddings. A positional embedding matrix has a size of SxD (where S is the max sequence length and D is the number of values in the embedding vector), which is manageable, but a word embedding matrix has a size of NxD (where N is the number of tokens in the vocabulary). Both N and D tend to be very large numbers (especially N), so we end up with incredibly demanding matrix operations (even memory and storage can begin to enter the realm of concern). As noted in the Computerphile video above, Word2Vec actually learns word embeddings quite efficiently, so using such a technique to pre-learn word embeddings could, likely at the cost of model/embedding quality, greatly reduce computational requirements.

It's worth concluding that we end up at an interesting junction: learning positional embeddings directly isn't very demanding, but it also doesn't provide much of a quality advantage; learning word embeddings directly _is_ very demanding, but it also _does_ provide a quality advantage.
