# Module 8 Short Answers

Read a summary of the findings from Google DeepMind's 2022 Chinchilla paper, which was a turning point for the field in how we scale our models. If you are interested in reading more, the full paper is here (optional): https://arxiv.org/abs/2203.15556

Read this blog post:

https://www.alignmentforum.org/posts/6Fpvch8RR29qLEWNH/chinchilla-s-wild-implications

## Question 1

What is a "scaling law" and how is it useful when designing LLMs?

**Answer**

The scaling law discussed in the blog post above can best be summarized as an equation that calculates the loss value an LLM will reach given the LLM's parameter count and the number of tokens in the dataset it is trained on. As the magnitude of these two values increases, the loss decreases, approaching an irreducible loss term. This law has clear utility for designing LLMs; most notably, it allows us to project the comparative performance of a proposed LLM against the performance of existing models. Per the topic of this module, LLMs require great scale (with respect to both parameter count and dataset size) to exhibit effective performance, and accommodating this scale is very cost- and time-intensive. Thus, being able to predict the level of performance you should get from the incredibly high investment of training an LLM is key.

## Question 2

What is the key insight from Chinchilla's scaling laws compared to previous work?

**Answer**

The Chinchilla model demonstrates a very important principle about LLM training: dataset size has an outsized impact on model performance compared to parameter count. In other words, the scaling law that governs loss, which is a function of dataset size and parameter count, decreases considerably more when dataset size is increased rather than parameter count. Increasing parameter count has _some_ impact on lowering loss, but this impact is not especially worthwhile. Previous models, such as GPT-3, kept dataset size roughly the same; per the blog post above, "until [the Chinchilla] paper, it was conventional to train all large LMs on roughly 300B tokens of data." Efforts were instead concentrated on parameter count. Since Chinchilla's 2022 debut, though, increasing dataset size has become a focal point.
