# Introduction
In this laboratory we will get our hands dirty working with Large Language Models (e.g. GPT and BERT) to do various useful things. I you haven't already, it is highly recommended to:

+ Read the [Attention is All you Need](https://arxiv.org/abs/1706.03762) paper, which is the basis for all transformer-based LLMs.
+ Watch (and potentially *code along*) with this [Andrej Karpathy video](https://www.youtube.com/watch?v=kCc8FmEb1nY) which shows you how to build an autoregressive GPT model from the ground up.

# Exercise 1: Warming Up
In this first exercise you will train a *small* autoregressive GPT model for character generation (the one used by Karpathy in his video) to generate text in the style of Dante Aligheri. Use [this file](https://archive.org/stream/ladivinacommedia00997gut/1ddcd09.txt), which contains the entire text of Dante's Inferno (**note**: you will have to delete some introductory text at the top of the file before training). Train the model for a few epochs, monitor the loss, and generate some text at the end of training. Qualitatively evaluate the results.

# Exercise 2: Working with Real LLMs

Our toy GPT can only take us so far. In this exercise we will see how to use the [Hugging Face](https://huggingface.co/) model and dataset ecosystem to access a *huge* variety of pre-trained transformer models.

## Exercise 2.1: Installation and text tokenization

First things first, we need to install the [Hugging Face transformer library](https://huggingface.co/docs/transformers/index):

    conda install -c huggingface -c conda-forge transformers
    
The key classes that you will work with are `GPT2Tokenizer` to encode text into sub-word tokens, and the `GPT2LMHeadModel`. **Note** the `LMHead` part of the class name -- this is the version of the GPT2 architecture that has the text prediction heads attached to the final hidden layer representations (i.e. what we need to **generate** text). 

Instantiate the `GPT2Tokenizer` and experiment with encoding text into integer tokens. Compare the length of input with the encoded sequence length.

**Tip**: Pass the `return_tensors='pt'` argument to the tokenizer to get Pytorch tensors as output (instead of lists).

This GPT2 tokenizer works on a subword level, so what I could notice is that, tipically, inputs are divided into several 2/3/4 words chunks and encoded to a particular integer.
As the input string sequences increases their length, the encoded sequence length increases.
Input sequences of 2/3/4 characters can be encoded to a single integer.

While inputing the sequence *"Hello World"*, I could notice that the tokenizer has a single integer for the whole "Hello" and "World" words, suggesting that many existing english (and not only, maybe) words are encoded to a single integer.

Passing the `return_tensors = 'pt'` argument makes the tokenizer output PyTorch tensors instead of lists.

From the original [Hugging Face Documentation](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer), we can read that:
> This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will be encoded differently whether it is at the beginning of the sentence (without space) or not.

Trying to see the behaviour of slightly different versions of the sequence *"Hello World"*, it's possible to notice that:
- *"Hello World"* encodes to tensor([[15496,  2159]])
- *"hello World"* encodes to tensor([[31373,  2159]]) --> Case matters
- *" hello World"* encodes to tensor([[23748,  2159]]) --> Space matters!
- *"HelloWorld"* encodes to tensor([[15496, 10603]])
- *"Hello World "* encodes to tensor([[15496,  2159,   220]]) --> But space character has its own encoding integer when nothing follows it

## Exercise 2.2: Generating Text

There are a lot of ways we can, given a *prompt* in input, sample text from a GPT2 model. Instantiate a pre-trained `GPT2LMHeadModel` and use the [`generate()`](https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) method to generate text from a prompt.

**Note**: The default inference mode for GPT2 is *greedy* which might not result in satisfying generated text. Look at the `do_sample` and `temperature` parameters.

In order to qualitatively evaluate the performance of the `generate()` function and the effect of its arguments, I decided to fix the text prompt to be the same at every run: *"Who knows if God exists, but for sure I"*

- `do_sample = False, temperature = 1.0`: with no argument tuned, the generation is totally greedy and helds no sense. The text generated is a simple sentence of text repeated over and over until the `max_new_tokens` limit of tokens is reached. When `do_sample` is `False`, it's like having a very low `temperature` value, as sampling (= source of noise) is frozen in favour of a deterministic, greedy approach to generation.
- Setting `do_sample` to `True` unlocks the generation, by allowing sampling of more diverse original sequences of tokens, instead of giving always the same greedy text. The tuning of `temperature` lets the magic happen!
- The higher the `temperature`, the "noisier" and unpredictable the generation will be. Very high temperatures lead to, again, non-sense generation, with wrong words and sequences of random symbols.

# Exercise 3: Reusing Pre-trained LLMs (choose one)

Choose **one** of the following exercises (well, *at least* one). In each of these you are asked to adapt a pre-trained LLM (`GPT2Model` or `DistilBERT` are two good choices) to a new Natural Language Understanding task. A few comments:

+ Since GPT2 is a *autoregressive* model, there is no latent space aggregation at the last transformer layer (you get the same number of tokens out that you give in input). To use a pre-trained model for a classification or retrieval task, you should aggregate these tokens somehow (or opportunistically select *one* to use).

+ BERT models (including DistilBERT) have a special [CLS] token prepended to each latent representation in output from a self-attention block. You can directly use this as a representation for classification (or retrieval).

+ The first *two* exercises below can probably be done *without* any fine-tuning - that is, just training a shallow MLP to classify or represent with the appropriate loss function.

# Exercise 3.1: Training a Text Classifier (easy)

Peruse the [text classification datasets on Hugging Face](https://huggingface.co/datasets?task_categories=task_categories:text-classification&sort=downloads). Choose a *moderately* sized dataset and use a LLM to train a classifier to solve the problem.

**Note**: A good first baseline for this problem is certainly to use an LLM *exclusively* as a feature extractor and then train a shallow model.

So far, I just tried to fine-tune a pretrained DistilBERT model for the Sequence Classification task.

But for this specific exercise, fine-tuning can be avoided! One hint is to use DistilBERT *only* as a mere feature extractor, and to use a very shallow model (an MLP, or even a Logistic Regression!) on the final representation for the multi-class classification.

Let's try to do so!

The choice of the dataset is crucial for what we're trying to achieve: text classification without having to fine-tune DistilBERT.

Looking at the Hugging Face Datasets, one of the best datasets to use is the *ag_news*: a moderately sized multi-class dataset, with perfectly balanced classes.
But, as always, I cannot feel satisfied with easy things: my attention got captured by the *dair-ai/emotion* dataset too, that, in comparison to the previous one, looks like a real mess! 6 classes, skewed, with not that much data available.

I'll try to face the same challenge using the two datasets, in order to check and report any difference encountered.

Looking at the DistilBertTokenizer, I can see that it's a word level tokenizer, at least for the english language.

- Using the *dair-ai/emotion* dataset, the multi-class classifier, with or without any tweaks on the loss weights to address the class imbalance problem, struggles to reach 67% Test Accuracy (while a fine-tuned DistilBERT is capable to go up to 94% accuracy).
- Using the *ag_news* dataset, instead, the classifier makes no effort to provide results with 92% Test Accuracy.

Furthermore, results obtained by training a simple MLP and an even simpler Logistic Regression are basically the same.

# Exercise 3.2: Training a Question Answering Model (harder)

Peruse the [multiple choice question answering datasets on Hugging Face](https://huggingface.co/datasets?task_categories=task_categories:multiple-choice&sort=downloads). Chose a *moderately* sized one and train a model to answer contextualized multiple-choice questions. You *might* be able to avoid fine-tuning by training a simple model to *rank* the multiple choices (see margin ranking loss in Pytorch).

# Exercise 3.3: Training a Retrieval Model (hardest)

The Hugging Face dataset repository contains a large number of ["text retrieval" problems](https://huggingface.co/datasets?task_categories=task_categories:text-retrieval&p=1&sort=downloads). These tasks generally require that the model measure *similarity* between text in some metric space -- naively, just a cosine similarity between [CLS] tokens can get you pretty far. Find an interesting retrieval problem and train a model (starting from a pre-trained LLM of course) to solve it.

**Tip**: Sometimes identifying the *retrieval* problems in these datasets can be half the challenge. [This dataset](https://huggingface.co/datasets/BeIR/scifact) might be a good starting point.
