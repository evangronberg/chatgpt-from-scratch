# Module 6 Short Answers

Read the beginning sections of two seminal works in LLMs: GPT-2 and T5. Read sections 1 and 2 of the GPT-2 paper:

https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

Also read sections 1 and 2 of the T5 paper:

https://arxiv.org/pdf/1910.10683

## Question 1

GPT-2 and T5 introduced similar ideas in the beginning of 2019. The key idea is stated in the GPT-2 paper: "[T]he supervised objective is the same as the unsupervised objective, but only evaluated on a subset of the sequence." Explain what this means in your own words.

**Answer**

Per this module's programming assignment, training a GPT model is something of a supervised-unsupervised hybrid approach. That is, the corpus the model is trained on is not formally labeled, but we use the sequential nature of text to effectively create labels. An explicitly supervised approach pairs input text with an ideal output (e.g., a question with an answer). The GPT training approach, however, described by Radford et al. as an unsupervised approach, has the model predict next words across the entire corpus. The quote above, then, is best interpreted as saying that a more traditional, supervised approach only evaluates model performance against explicit labels, while the unsupervised approach evaluates model performance against next-word prediction across the full dataset.
