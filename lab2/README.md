# Deep Learning Applications: Laboratory #2 - LLMs

In this Laboratory I will get to work with Large Language Models (e.g. GPT and BERT).
Substantial part of this Lab relies on the notorious paper [Attention is All you Need](https://arxiv.org/abs/1706.03762), which introduced the attention-based Transformer architecture and is the basis for all Transformer-based LLMs.

In order to be able to understand more deeply the logic behind Language Modeling and language models, I coded along with [this video](https://www.youtube.com/watch?v=kCc8FmEb1nY) made by Andrej Karpathy, a tutorial for building an autoregressive GPT language model from scratch. The code I produced will be very useful for the entire Lab.

## Exercise 1: Warming Up
In this first exercise I will train a *small* autoregressive GPT model for character generation, the exact same of the Karpathy video, to generate text in a particular style.
I was firstly asked to generate text in the style of Dante Alighieri using [this file](https://archive.org/stream/ladivinacommedia00997gut/1ddcd09.txt), which contains the entire text of Dante's Divine Comedy. After deleting some introductory text at the top of the file and problematic characters, I was able to start training.
I trained the model for 10 epochs, and generated some text at the end of training.
As perfectly explained in the Karpathy video, monitoring the loss can give vital information about how the learning process is going.

**qualitative evaluation**

Loss established, in the end, at 2.00 (numero a caso), which means that there's still a lot of room for improvement.
However, it must be said that the model generated quite good quality text despite a relatively small dataset, even if there are quite many gramatical errors and it completely lacks of semantics and meaning. Text clearly was generated without taking into account context, so basically, there's still a lot of work to do.

Before going on to the next exercise, I decided to try my language model on another dataset, in order to check how well it can handle different languages: for this reason, I trained the same model on a dataset containing the lyrics of all Taylor Swift's discography, and was curious about the results.

(potrei provare al volo verso la fine a fare training su dataset con dntro entrambe le cose, quindi due lingue allo stesso momento per vedere se sminchia)

---
## Exercise 2: Working with Real LLMs
As previously seen, my toy GPT can only take me this far. Language Modeling is a field in which models need a real *ton* of text data of any kind and topic in order to generate meaningful text. So, small companies and researchers or students must start using **pre-trained** Language Models.
Luckily, the [HuggingFace](https://huggingface.co/) ecosystem allows to access a *huge* variety of pre-trained Transformer models and datasets for any kind of application. In this exercise, I will learn to use stuff from HuggingFace for my future Language Modeling projects.

### Exercise 2.1: Installation and text tokenization
First things first, I need to install the [HuggingFace Transformer library](https://huggingface.co/docs/transformers/index) on my `conda` environment (which is the same used in the first Laboratory, **DLA**).

The language model I will focus on is **GPT2**. The key classes that I will work with are `GPT2Tokenizer` to encode text into sub-word tokens, and the `GPT2LMHeadModel`, a version of GPT2 that has *text prediction heads* attached to the final hidden layer representations (i.e. what we need to **generate** text). 

This GPT2 tokenizer works on a subword level. Comparing the length of input with the encoded sequence length, I could notice that inputs are typically divided into several 2/3/4 words chunks and encoded to a particular integer.
As the input string sequences increases their length, the encoded sequence length increases. Input sequences of 2/3/4 characters can be encoded to a single integer.

While inputing the sequence *'Hello World'*, I could notice that the tokenizer has a single integer for the whole "Hello" and "World" words, suggesting that many existing english (and not only, maybe) words are encoded to a single integer.

Passing the `return_tensors = 'pt'` argument to the instantiated `GPT2Tokenizer` makes the tokenizer output PyTorch tensors instead of lists.

From the official [HuggingFace Documentation](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer), we can read that:
> This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will be encoded differently whether it is at the beginning of the sentence (without space) or not.

Let's try to see how slightly different versions of the same sequence *"Hello World"* are addressed by the tokenizer:
- *"Hello World"* encodes to tensor([[15496, 2159]])
- *"hello World"* encodes to tensor([[31373, 2159]]) --> Case matters
- *" hello World"* encodes to tensor([[23748, 2159]]) --> Space matters!
- *"HelloWorld"* encodes to tensor([[15496, 10603]]) --> Again, space matters
- *"Hello World "* encodes to tensor([[15496, 2159, 220]]) --> But space character has its own encoding integer when nothing follows it

### Exercise 2.2: Generating Text
Given a particular input *prompt*, there's a lot of ways we can sample text from a GPT2 model. Let's try this: I will instantiate a pre-trained `GPT2LMHeadModel` and use the [`generate()`](https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) method to generate text from a prompt.

It's important to note that the default inference mode for GPT2 is *greedy*, which might not result in satisfying generated text. Yet, the `generate()` function has some useful parameters that can be tuned for generation customization, like `do_sample` and `temperature`.
In order to qualitatively evaluate the performance of the `generate()` function and the effect of its arguments, I fixed the text prompt to be the same at every run: *"Who knows if God exists, but for sure I "* ...I know, very philosophical.

- `do_sample = False, temperature = 1.0`: this is the default `generate()` function, which gives *greedy* generation and helds little to no sense. The text generated is a simple sentence of text repeated over and over until the `max_new_tokens` limit of tokens is reached. When `do_sample` is `False`, it's like having a very low `temperature` value, as sampling (= source of noise) is frozen in favour of a deterministic, greedy approach to generation.
- Setting `do_sample = True` unlocks the generation, by allowing sampling of more diverse original sequences of tokens, instead of giving always the same greedy text.
- But tuning `temperature` lets the real magic happen! The higher the `temperature`, the "noisier" and more unpredictable the generation will be. Very high temperatures lead to, again, non-sense generation, with nonsensical words and sequences of random symbols.

---

## Exercise 3: Reusing Pre-trained LLMs
In each of the following exercises I'm asked to adapt a pre-trained LLM to a new Natural Language Understanding task.
I chose to use `DistilBERT` in order to better understand the functioning of BERT models.
BERT models (including DistilBERT) have a special [CLS] token prepended to each latent representation in output from a self-attention block. I will directly use this as a representation for addressing my following tasks.

Note: Exercises 3.1 and 3.2 can be done *without* any fine-tuning - that is, just training a shallow MLP to classify or represent with the appropriate loss function. So, that's what I tried to do.

### Exercise 3.1: Training a Text Classifier
An important choice is the dataset that will be used for the task. Looking at the [text classification datasets on HuggingFace](https://huggingface.co/datasets?task_categories=task_categories:text-classification&sort=downloads), one of the best datasets to use is the *ag_news*: a moderately sized multi-class dataset about news from all over the world, with 4 perfectly balanced classes.
But, as always, I can't feel satisfied with easy things: my attention got captured by the *dair-ai/emotion* dataset too, a set about sentiment of snippets of text that, in comparison to the previous one, looks like a real mess! 6 classes, skewed, with not so much data available.
I'll compare the two datasets, check and report any difference encountered in order to have a more complete view of how my Language Model works with different kinds of data.



So far, I just tried to fine-tune a pretrained DistilBERT model for the Sequence Classification task.

But for this specific exercise, fine-tuning can be avoided! One hint is to use DistilBERT *only* as a feature extractor, and to use a very shallow model (an MLP, or even a Logistic Regression!) on the final representation for the multi-class classification.

Looking at the DistilBertTokenizer, I can see that it's a **word level** tokenizer, at least for the english language.

- Using the *dair-ai/emotion* dataset, the multi-class classifier, with or without any tweaks on the loss weights to address the class imbalance problem, struggles to reach 67% Test Accuracy (while a fine-tuned DistilBERT is capable to go up to 94% accuracy).
- Using the *ag_news* dataset, instead, the classifier makes no effort to provide results with 92% Test Accuracy.

Furthermore, results obtained by training a simple MLP and an even simpler Logistic Regression are basically the same.

---

### Exercise 3.2: Training a Question Answering Model
The Question-Answering task consists in predicting and giving the correct answer at particular contextualized multiple-choice questions.
To address it, I will pick one dataset from the [multiple choice question answering datasets on HuggingFace](https://huggingface.co/datasets?task_categories=task_categories:multiple-choice&sort=downloads). Even here, for computation purposes, it is good to choose a *moderately* sized one.
Here, I *might* be able to avoid fine-tuning by training a simple model to *rank* the multiple choices using *margin loss* in PyTorch.

---

### Exercise 3.3: Training a Retrieval Model
The HuggingFace dataset repository contains a large number of ["text retrieval" problems](https://huggingface.co/datasets?task_categories=task_categories:text-retrieval&p=1&sort=downloads). These tasks generally require that the model measure *similarity* between text in some metric space.
Naively, just a *cosine similarity* between [CLS] tokens could get me pretty far.

I chose the text retrieval problem.

**Tip**: Sometimes identifying the *retrieval* problems in these datasets can be half the challenge. [This dataset](https://huggingface.co/datasets/BeIR/scifact) might be a good starting point.
