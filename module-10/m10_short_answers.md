# Module 10 Short Answers

Read the LoRA paper, which presents possibly the most widely used PEFT method:

https://arxiv.org/pdf/2106.09685

## Question 1

LoRA claims several "key advantages" for different tasks. How does LoRA improve over SFT when multiple tasks need to be handled by LLMs?

**Answer**

The clearest advantage of LoRA over SFT is found in its training efficiency. However, LoRA also provides a clear advantage over SFT for task switching. SFT takes a model and fine-tunes _all_ parameters; in other words, we are left with one model for one use case. Switching to another use case, then, becomes a matter of switching to another model; most notably, this means that we will be required to store and load multiple models in and out, each of which is very large. LoRA enables a single model to be used for a variety of tasks, only swapping between low-rank matrices (i.e., matrices requiring little storage) when the use case changes.

## Question 2

In your opinion, how scalable is this idea? What are some applications which may be made possible with LoRA, and which might still be unattainable?

**Answer**

LoRA seems to me to be a very scalable idea; due to the incredibly low rank values that are chosen for the algorithm, LoRA essentially scales linearly. Furthermore, it seems that the greater the scale of the model it integrates with, the more pronounced its value becomes.

The scalability of LoRA lends itself toward new applications. Organizations smaller than the likes of OpenAI and Google gain the ability to fine-tune models of more substantial, respectable size. That said, it is worth noting that, per last module's lecture, fine-tuning is more about teaching a model a pattern of _behavior_ rather than direct knowledge. The latter type of use case tends to be better served by RAG, but fine-tuning still has its use for these smaller organizations. My personal understanding is that model performance within specific professional domains like law and healthcare benefit from a domain-specific fine-tuned model. Furthermore, the ability of LoRA to easily swap between different sets of low-rank matrices also enables smaller organizations to more easily serve a variety of use cases while only having to deploy a single model (e.g., this could mean a single model that performs well at summarization, idea generation, and question answering all in one).

Of course, LoRA, like any fine-tuning mechanism, still requires a significant amount of training data, and this is clearly a limiting factor for many smaller organizations' use cases. In other words, LoRA still can't overcome a lack of data, and so building fine-tuned applications in contexts in which data is sparse is still unattainable.
