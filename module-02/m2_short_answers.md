# Module 2 Short Answers

Watch the Computerphile video on Unicode:

https://www.youtube.com/watch?v=MijmeoH9LT4

Also watch an overview of byte-pair encoding (BPE) from HuggingFace:

https://www.youtube.com/watch?v=HEikzVL-lZU

## Question 1

Explain how a byte-level BPE is able to tokenize a character it has never seen before (i.e., an emoji).

**Answer**

While in this module's assignment we apply the BPE algorithm to characters/strings, BPE may be applied to sequences of any sort – including byte sequences. Bytes are what comprise the UTF-8 character encoding standard discussed in the Computerphile video above, and per that video, UTF-8 encompasses upwards of 100,000 characters. It is therefore quite possible for BPE to come across a new UTF-8 character it hasn't seen before. If this happens while BPE is tokenizing at the byte level, the algorithm is able to draw upon the byte-level tokens it has already collected in its training process – in fact, for byte-level tokenization, the initial vocabularly will contain all possible bytes anyway. The tokenizer won't output a token encompassing all of that previously unseen character's bytes, but it will nonetheless be able to tokenize the character.
