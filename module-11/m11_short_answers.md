# Module 11 Short Answers

Watch an excellent video about the general recipe for training a model like ChatGPT (from mid-2023). It is presented by Andrej Karpathy, co-founder of OpenAI and former director of AI at Tesla.

https://www.youtube.com/watch?v=bZQun8Y4L2A

## Question 1

Why is SFT included in the RLHF pipeline? (i.e., why can't we just start with RL?)

**Answer**

SFT is responsible for training a model to exhibit particular behavior (i.e., it turns a text continuation model into one that actually provides helpful, use-case-specific responses). Per Karpathy's talk, an RLHF pipeline requires a reward model, and this model is trained on response _comparisons_. That is, it is trained on sets of responses that have been ranked by humans in order to determine which responses humans like best (i.e., which responses receive the highest "reward"). For these responses to be produced in the first place though, the core language model at play here must have been taught the proper response behavior pattern via SFT. Thus, without SFT, we cannot effectively train RLHF's reward model.

## Question 2

What are some potential pitfalls when building and using a reward model? What are the downstream impacts on model alignment?

**Answer**

The process of training a reward model relies upon human judgment in ranking LLM responses. Furthermore, this process is done at scale (100K-1M comparisons per Karpathy's talk). While seemingly reputable organizations do exist for getting this job done, it's still incredibly difficult (if not impossible) to verify that these comparisons have not been done poorly, whether intentionally or intentionally. Based on my use of ChatGPT, these comparisons also seem to crowd-sourced to some degree; this is even more obvious cause for concern with respect to labeling quality.

Taking a step back, the mechanism by which the reward model modifies model behavior (i.e., weighting rewarded tokens more highly) seems like a particular source of risk when labeling is crowd-sourced. Any user could purposefully prompt the model to output distasteful content, and if asked to rank responses, the user could more highly favor the most distasteful of the responses. It is hard to say how great an effect the phenomenon could have on model alignment, but it at least seems worth addressing.
