# Module 9 Short Answers

Read about Alpaca, a fine-tuned version of LLaMA that showed incredible performance gains while using a small amount of resources.

Read this overview from the Alpaca authors:

https://crfm.stanford.edu/2023/03/13/alpaca.html

## Question 1

An interesting aspect of Alpaca is that it was fine-tuned on synthetic data. What are some of the possible benefits and shortcomings of this approach?

**Answer**

The general idea of fine-tuning models with synthetic data is interesting, but I find Alpaca's approach in particular to be fascinating: to mimic GPT-level behavior, Alpaca actually uses _GPT-generated_ examples. An obvious benefit of this approach is having an existing larger model "teach" a smaller model such that it approximates the larger model's performance and behavior while also being less demanding. Another benefit is also clear – such data may be easily obtained since it does not need to be sourced from human-written content.

A shortcoming of this method is the questionable verifiability of synthetic data – simply put, it's hard to know for sure that, at scale, what a model has produced is natural and high-quality. That isn't to say that all human-written content is guaranteed to be good (it certainly isn't), but at least _certain_ types of human-written content are effectively guaranteed to meet high standards (e.g., textbooks and books from reputable publishers). For a dataset of 52K instruction-following examples (as is the case for Alpaca), it is simply not feasible to look at every example and verify its "natural-ness," whereas human-written content is inherently natural.

## Question 2

From an ethical standpoint, is training/tuning on the outputs of another model (that is not your work) a valid approach to building a new model? There is not a correct answer here – just looking for your own opinion.

**Answer**

This is a broad question that resists a simple answer. In the case of Alpaca, I believe that Taori et al. take an ethical approach, although this arguably the "trivial" ethical approach – simply intending the model "only for academic research" and prohibiting "any commercial use." By introducing this legal limit, Taori et al. arguable extricate themselves from any moral culpability for the misbehavior of the model – it simply exists to further human knowledge.

Suppose, however, that Alpaca _was_ usable in a commercial context, or was perhaps even a closed-source model developed directly for a commercial use case. We would, of course, run into legal problems with Alpaca specifically – it was trained on OpenAI-produced data, and OpenAI "prohibit[s] developing models that compete with OpenAI." This would be an ethical problem in that Alpaca would be violating the terms of use it agrees upon with OpenAI, but frankly, if OpenAI did not include this clause in its terms of use, then I see no ethical issue. The real ethical issues tend to crop up on the side of those like OpenAI – it is up for serious debate whether or not they should be able to leverage the human-produced data (i.e., human creativity) they often use for pre-training. 
