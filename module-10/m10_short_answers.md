# Module 10 Short Answers

Read the LoRA paper, which presents possibly the most widely used PEFT method:

https://arxiv.org/pdf/2106.09685

## Question 1

LoRA claims several "key advantages" for different tasks. How does LoRA improve over SFT when multiple tasks need to be handled by LLMs?

**Answer**

The clearest advantage of LoRA over SFT is found in its training efficiency. However, LoRAD also provides a clear advantage over SFT for task switching. SFT takes a model and fine-tunes _all_ parameters; in other words, we are left with one model for one use case. Switching to another use case, then, becomes a matter of switching to another model; most notably, this means that we will be required to store and load multiple models in and out, each of which is very large. LoRA enables a single model to be used for a variety of tasks, only swapping between low-rank matrices (i.e., matrices requiring little storage) when the use case changes.

## Question 2

In your opinion, how scalable is this idea? What are some applications which may be made possible with LoRA, and which might still be unattainable?

**Answer**

<!-- LoRA seems to me to be a very scalable idea; in fact, it seems that the greater the scale of the model it integrates with, the more pronounced its value becomes. The range of organizations that are now able to fine-tune, and therefore better leverage, LLMs increases notably. An application that I see LoRA as particularly useful for would be fine-tuning an LLM for  -->
