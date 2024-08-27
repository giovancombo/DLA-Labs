# Deep Learning Applications: Laboratory #2 - LLMs

In this Laboratory I will get to work with Large Language Models (e.g. GPT and BERT).
Substantial part of this Lab relies on the notorious paper [Attention is All you Need](https://arxiv.org/abs/1706.03762), which introduced the attention-based Transformer architecture and is the basis for all Transformer-based LLMs.

In order to be able to understand more deeply the logic behind Language Modeling and language models, I coded along with [this video](https://www.youtube.com/watch?v=kCc8FmEb1nY) made by Andrej Karpathy, a tutorial for building an autoregressive GPT language model from scratch. The code I produced will be very useful for the entire Lab.

## Exercise 1: Warming Up
In this first exercise I will train a *small* autoregressive GPT model for character generation, the exact same of Karpathy's video, to generate text in a particular style.

I was firstly asked to generate text in the style of Dante Alighieri using [this file](https://archive.org/stream/ladivinacommedia00997gut/1ddcd09.txt) (data at `data/divina_commedia.txt`), which contains the entire text of Dante's Divine Comedy. After removing some introductory text at the top of the file and problematic characters, I was able to start training. Here the training settings I used:

+ **5000** steps of training
+ Batch size: **32** and **256**
+ Block size: **64** and **128**
+ Learning rate: **5e-4**
+ Dropout rate: **0.2**
+ Token Embedding dimension: comparison between **32**, **128** and **256**

Each model I trained is saved in the `1_models` folder, and comes with a `generation_training.txt` file that contains some text generated every 100 steps by that model. This allowed me to appreciate how text generation progressively improves throughout the steps.

After training my model on *italian* Dante's text, I was curious about training it on an *english* corpus too: I decided to use the lyrics of all songs by Taylor Swift (data at `data/taylor_swift.txt`), so that I could even check if the text generated could vaguely resemble the lyrics of a real song.

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab2/images/1_models_valloss.png" width="49%">
</p>
<p align="center"><i><b>Figure 1</b> | Losses of the models trained on the two datasets: Divina Commedia (green) and Taylor Swift (orange)</i></p>

*Figure 1* shows how the **Token Embedding dimension** has a particular influence on the ability of the model in capturing the complexity of the datasets. And this difference between models trained with **32**- and **128**-dimensional embeddings can be noticed also with a qualitatively evaluation of the text generated. Let's take a look at those `generation_training.txt` files:

|*Step*|*Divina Commedia, 32-dim Embedding*|*Divina Commedia, 128-dim Embedding*|
|:---:|:---:|:---:|
|100|om Tli ld fuieloattovG dhVa  os  ee Qrg ex, cednn s: nt.i me q Po 'ha i i rfte se'o cem cga llo arre tednee -n,adXeRrta sro eGi,Ri Ue ce lon-gEL l  ini co eroioiuaode 'teIre lamo T faemihiaire m<br><br>*Loss: 2.9188*|fe  fun da ndRm'lo che du  "er g'ise toi galome ' cpre di' po guiverne cha ratretia, al a, E i  custti, Betr  r pra a meso darante sttttisterismi le sposcrr ntato mer- che l ideNo; a' esa, no fron <br><br>*Loss: 2.3669*|
|1500|se mamban viuii ma pan eltregegise cola' de' manaba m'a e che vono, cheia na meco cosalue cca l cosacin ancol'uno sana l'a,din su fiseve, pcoscu e mosimo sine,  la bolal liva a li pe.  Milorai pe'<br><br>*Loss: 2.2332*|mi fosse al primo, fosso discerso e dolpa chiolve a dimando grizio! <<Ghe disfetterno indicuro con giorno, e terro 'l prommo da' vermo danno, di fuor di quella mal mezzore in susoce; e quan<br><br>*Loss: 1.5807*|
|3000|rol segna fer rarengia binostacr sa dan masce l, suanantre nar cengriuco suro e l quegoce, ria <Itor che ovin ge sar 'le si ia stesca. Alve giala lel qua cci e ver cermpa, por ren lali fun<br><br>*Loss: 2.0851*|ben cantano in quest'anrosche. In se' fu' ch'i' nel temo mi fele; canti bollo, orata di sapi pesa; ma piu' che tu forse coi tormente; ch'e' non vediti cio' mi vuol chi sonse, ne' moschi se' 'l<br><br>*Loss: 1.5258*|
|4900|o dacche di torran divei>>. Noschizi tuscuo mantovo sui ra` le mepper Sa tu tuinuisi, gadio magnuallo inda; qua mostro, <<Ontuo suescede liagpie, e limanta disto che near peschio sa a salta lua<br><br>*Loss: 1.9509*|e Calcavalier lo 'nferno in quella piu' nacque ch'incrocciar con le paura e diserrole, qual che fosse il muro de la tua fiso, che confesso i pasto e piu` posa, pero' ch'un sommo tra 'l piant<br><br>*Loss: 1.5380*|

<p align="center"><i><b>Table 1</b> | Progression of text generation of 200 tokens using two different Embedding dimensions on the Divina Commedia dataset</i></p><br>

|*Step*|*Taylor Swift, 32-dim Embedding*|*Taylor Swift, 128-dim Embedding*|
|:---:|:---:|:---:|
|100|o  ,em, teIend ooa X Ipl slmNvim  s “Xu–Wi hmmch[<br>rts oaeat w—k5 w?oQl,c roon n la K​a wehdkgtln-,ssb eeLa  li ao  trYey]aemda! bi;I0m,d<br>Ani g lteluoo h iTs t triyYe ren RbCe(.k еdoss d o Tdm  Q b S<br><br>*Loss: 3.1831*|Koould, uve'lle wonnd th wr owham or w I'mer co Thoousessald 6or rouse ayl<br>[Ct eranere-Che An uortest erres])<br>[Shol st yorst loheneuran yighim<br>Aneve I you' c iq coeeou<br>Thewhon ing ct bu y wero y Byv<br><br>*Loss: 2.4845*|
|1500|I've menibu I olds f outhealike be<br>Andes alsont'malssevee<br>Ash, ifeerer n, I' mecepe]<br>I watowhare ig I-deringugengoft e th"Oas<br>Lat it yo (Sait me (He eve hancaaime s’mt and ar ngan wea bet neatst yond<br><br>*Loss: 2.3532*|With your feels mind<br>Ail I catcheres you were words<br>You don't know what I was lup in the scar?<br>Are you just the we stay<br>I never heel the sky?<br>Just brokers for your night<br>Something what'se, love you mi<br><br>*Loss: 1.6168*|
|3000|Whe ar wandy, warit totna ndig<br>The I cesthe I brso thaint at's mit freate of heme head wor oul<br>I merednnd ous and wo deemtho I yoren, she drcemow, an'tin, ghas's<br>I olof wan die bewaun tt'men w ca was<br><br>*Loss: 2.1478*|And you what a mall placion<br>And you know places in into mind-seide<br>And you don't know that<br>Forgo that was red, right<br>I would feel wond[Verse 2]<br>You could see that this magic by this best<br>But the ones<br><br>*Loss: 1.6058*|
|4900|[Verse Thirs] Oh<br>She you wenou lingt hink you nory gray bed<br>ass't ait bacallling you're tought a nof ye[Choler 2]<br>You wald scaut driry, slak aone[Chorus]<br>You you and whant you<br>You'r the be golonat, sa<br><br>*Loss: 1.9927*|Heed for the ontowns in begin<br>Even if you're touching my mind<br>I heard to know his you were that fighting to you<br>You might also like[Pre-Chorus]<br>And we had you so go<br>Trake one who side joked a blue sin<br><br>*Loss: 1.6535*|

<p align="center"><i><b>Table 2</b> | Progression of text generation of 200 tokens using two different Embedding dimensions on the Taylor Swift dataset</i></p>

It must be said that the models generated quite good quality text despite having been trained on a relatively small dataset, even if there are quite many gramatical errors and it completely lacks of semantics and meaning. Text clearly was generated without taking much into account context, so basically, there's still room for improvement.

---

## Exercise 2: Working with Real LLMs
As previously seen, my toy GPT can only take me this far. Language Modeling is a field in which models need a real *ton* of text data of any kind and topic in order to generate meaningful text. So, small companies and researchers or students must start using **pre-trained** Language Models.
Luckily, the [HuggingFace](https://huggingface.co/) ecosystem allows to access a *huge* variety of pre-trained Transformer models and datasets for any kind of application. In this exercise, I will learn to use stuff from HuggingFace for my future Language Modeling projects.

### Exercise 2.1: Installation and text tokenization
First things first, I need to install the [HuggingFace Transformer library](https://huggingface.co/docs/transformers/index) on my `conda` environment.

The language model I will focus on is **GPT2**. The key classes that I will work with are `GPT2Tokenizer` to encode text into sub-word tokens, and the `GPT2LMHeadModel`, a version of GPT2 that has *text prediction heads* attached to the final hidden layer representations (i.e., what we need to **generate** text).

While inputing the sequence *'Hello World'*, I could notice that the tokenizer has a single integer for the whole "Hello" and "World" words, suggesting that many existing english (and not only, maybe) words are encoded to a single integer.

From the official [HuggingFace Documentation](https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2Tokenizer), we can read that:
> This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will be encoded differently whether it is at the beginning of the sentence (without space) or not.

Let's try to see how slightly different versions of the same sequence *"Hello World"* are addressed by the tokenizer:
- *"Hello World"* encodes to tensor([[15496, 2159]])
- *"hello World"* encodes to tensor([[31373, 2159]]) --> Case matters
- *" hello World"* encodes to tensor([[23748, 2159]]) --> Space matters!
- *"HelloWorld"* encodes to tensor([[15496, 10603]]) --> Again, space matters
- *"Hello World "* encodes to tensor([[15496, 2159, 220]]) --> But space character has its own encoding integer when nothing follows it

### Exercise 2.2: Generating Text
Given a particular input *prompt*, there's a lot of ways we can sample text from a GPT2 model. I will instantiate a pre-trained `GPT2LMHeadModel` and use the [`generate()`](https://huggingface.co/docs/transformers/v4.27.2/en/main_classes/text_generation#transformers.GenerationMixin.generate) function to generate text from a prompt.

It's important to note that the default inference mode for GPT2 is *greedy*, which might not result in satisfying generated text. Yet, the `generate()` function has some useful parameters that can be tuned for generation customization, like `do_sample` and `temperature`.
In order to qualitatively evaluate the performance of the `generate()` function and the effect of its arguments, I fixed the text prompt to be the same at every run.

Here is some text generated tuning the function arguments (all outputs are available at `2_textgeneration/generation_logs.txt`):

**Prompt**: *"The main goal in life is "*

|*Function Arguments*|*Text Generated*|
|:---:|:---:|
|`do_sample = False`<br>`temperature = 1.0`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 0`|ee|
|`do_sample = True`<br>`temperature = 1.0`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 0`|unlocks the generation, by allowing sampling of more diverse original sequences of tokens, instead of giving always the same greedy text.|
|`do_sample = False`<br>`temperature = 1.5`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 0`|But tuning `temperature` lets the real magic happen! The higher the `temperature`, the "noisier" and more unpredictable the generation will be. Very high temperatures lead to, again, non-sense generation, with nonsensical words and sequences of random symbols.|
|`do_sample = False`<br>`temperature = 1.0`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 3`|But tuning `temperature` lets the real magic happen! The higher the `temperature`, the "noisier" and more unpredictable the generation will be. Very high temperatures lead to, again, non-sense generation, with nonsensical words and sequences of random symbols.|
|`do_sample = True`<br>`temperature = 1.0`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 0`|But tuning `temperature` lets the real magic happen! The higher the `temperature`, the "noisier" and more unpredictable the generation will be. Very high temperatures lead to, again, non-sense generation, with nonsensical words and sequences of random symbols.|
|`do_sample = False`<br>`temperature = 1.0`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 0`|But tuning `temperature` lets the real magic happen! The higher the `temperature`, the "noisier" and more unpredictable the generation will be. Very high temperatures lead to, again, non-sense generation, with nonsensical words and sequences of random symbols.|
|`do_sample = False`<br>`temperature = 1.0`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 0`|But tuning `temperature` lets the real magic happen! The higher the `temperature`, the "noisier" and more unpredictable the generation will be. Very high temperatures lead to, again, non-sense generation, with nonsensical words and sequences of random symbols.|
|`do_sample = False`<br>`temperature = 1.0`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 0`|But tuning `temperature` lets the real magic happen! The higher the `temperature`, the "noisier" and more unpredictable the generation will be. Very high temperatures lead to, again, non-sense generation, with nonsensical words and sequences of random symbols.|
|`do_sample = False`<br>`temperature = 1.0`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 0`|But tuning `temperature` lets the real magic happen! The higher the `temperature`, the "noisier" and more unpredictable the generation will be. Very high temperatures lead to, again, non-sense generation, with nonsensical words and sequences of random symbols.|
|`do_sample = False`<br>`temperature = 1.0`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 0`|But tuning `temperature` lets the real magic happen! The higher the `temperature`, the "noisier" and more unpredictable the generation will be. Very high temperatures lead to, again, non-sense generation, with nonsensical words and sequences of random symbols.|

The first one is the default `generate()` function, which gives *greedy* generation and helds little to no sense. The text generated is a simple sentence of text repeated over and over until the `max_new_tokens` limit of tokens is reached. When `do_sample` is `False`, it's like having a very low `temperature` value, as sampling (= source of noise) is frozen in favour of a deterministic, greedy approach to generation.

`do_sample = True` unlocks the generation, by allowing sampling of more diverse original sequences of tokens, instead of giving always the same greedy text.

But tuning `temperature` lets the real magic happen! The higher the `temperature`, the "noisier" and more unpredictable the generation will be. Very high temperatures lead to, again, non-sense generation, with nonsensical words and sequences of random symbols.

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
