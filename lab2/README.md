# Deep Learning Applications: Laboratory #2 - LLMs

In this Laboratory I will get to work with Large Language Models (e.g. GPT and BERT).
Substantial part of this Lab relies on the notorious paper [Attention is All you Need](https://arxiv.org/abs/1706.03762), which introduced the attention-based Transformer architecture and is the basis for all Transformer-based LLMs.

In order to be able to understand more deeply the logic behind Language Modeling and language models, I coded along with [this video](https://www.youtube.com/watch?v=kCc8FmEb1nY) made by Andrej Karpathy, a tutorial for building an autoregressive GPT language model from scratch. The code I produced will be very useful for the entire Lab.

## Exercise 1: Warming Up
In this first exercise, I will train a *small* autoregressive GPT model for character generation, identical to the one in Karpathy's video, to generate text in a particular style.

Initially, I was tasked with generating text in the style of Dante Alighieri using [this file](https://archive.org/stream/ladivinacommedia00997gut/1ddcd09.txt) (located at `data/divina_commedia.txt`), which contains the complete text of Dante's Divine Comedy. After removing some introductory text from the top of the file and problematic characters, I was able to begin training. Here are the training settings I used:

+ **5000** steps of training
+ Batch size: **32** and **256**
+ Block size: **64** and **128**
+ Learning rate: **5e-4**
+ Dropout rate: **0.2**
+ Token Embedding dimension: comparison between **32**, **128** and **256**

Each model I trained is saved in the `1_models` folder, accompanied with a `generation_training.txt` file.  This file contains samples of text generated every 100 steps by that model, allowing me to observe how text generation progressively improves throughout the training process.

After training my model on Dante's *Italian* text, I was curious about training it on an *English* corpus as well. I chose to use a collection of song lyrics by Taylor Swift (data at `data/taylor_swift.txt`). This selection allowed me to assess whether the generated text could vaguely resemble the style and structure of modern song lyrics.

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab2/images/1_models_valloss.png" width="49%">
</p>
<p align="center"><i><b>Figure 1</b> | Losses of the models trained on the two datasets: Divina Commedia (green) and Taylor Swift (orange)</i></p>

*Figure 1* demonstrates how the **Token Embedding dimension** significantly influences model's ability to capture the complexity of the datasets. This difference between models trained with **32**- and **128**-dimensional embeddings is also evident in a qualitative evaluation of the generated text. Let's take a look at the `generation_training.txt` files to observe these differences:

|*Step*|*Divina Commedia, 32-dim Embedding*|*Divina Commedia, 128-dim Embedding*|
|:---:|:---:|:---:|
|100|om Tli ld fuieloattovG dhVa  os  ee Qrg ex, cednn s: nt.i me q Po 'ha i i rfte se'o cem cga llo arre tednee -n,adXeRrta sro eGi,Ri Ue ce lon-gEL l  ini co eroioiuaode 'teIre lamo T faemihiaire m<br><br>*Loss: 2.9188*|fe  fun da ndRm'lo che du  "er g'ise toi galome ' cpre di' po guiverne cha ratretia, al a, E i  custti, Betr  r pra a meso darante sttttisterismi le sposcrr ntato mer- che l ideNo; a' esa, no fron <br><br>*Loss: 2.3669*|
|1500|se mamban viuii ma pan eltregegise cola' de' manaba m'a e che vono, cheia na meco cosalue cca l cosacin ancol'uno sana l'a,din su fiseve, pcoscu e mosimo sine,  la bolal liva a li pe.  Milorai pe'<br><br>*Loss: 2.2332*|mi fosse al primo, fosso discerso e dolpa chiolve a dimando grizio! <<Ghe disfetterno indicuro con giorno, e terro 'l prommo da' vermo danno, di fuor di quella mal mezzore in susoce; e quan<br><br>*Loss: 1.5807*|
|3000|rol segna fer rarengia binostacr sa dan masce l, suanantre nar cengriuco suro e l quegoce, ria <Itor che ovin ge sar 'le si ia stesca. Alve giala lel qua cci e ver cermpa, por ren lali fun<br><br>*Loss: 2.0851*|ben cantano in quest'anrosche. In se' fu' ch'i' nel temo mi fele; canti bollo, orata di sapi pesa; ma piu' che tu forse coi tormente; ch'e' non vediti cio' mi vuol chi sonse, ne' moschi se' 'l<br><br>*Loss: 1.5258*|
|4900|o dacche di torran divei>>. Noschizi tuscuo mantovo sui ra` le mepper Sa tu tuinuisi, gadio magnuallo inda; qua mostro, <<Ontuo suescede liagpie, e limanta disto che near peschio sa a salta lua<br><br>*Loss: 1.9509*|e Calcavalier lo 'nferno in quella piu' nacque ch'incrocciar con le paura e diserrole, qual che fosse il muro de la tua fiso, che confesso i pasto e piu` posa, pero' ch'un sommo tra 'l piant<br><br>*Loss: 1.5380*|

<p align="center"><i><b>Table 1</b> | Progression of text generation (200 tokens) at different training steps using two Embedding dimensions on the Divina Commedia dataset</i></p><br>

|*Step*|*Taylor Swift, 32-dim Embedding*|*Taylor Swift, 128-dim Embedding*|
|:---:|:---:|:---:|
|100|o  ,em, teIend ooa X Ipl slmNvim  s “Xu–Wi hmmch[<br>rts oaeat w—k5 w?oQl,c roon n la K​a wehdkgtln-,ssb eeLa  li ao  trYey]aemda! bi;I0m,d<br>Ani g lteluoo h iTs t triyYe ren RbCe(.k еdoss d o Tdm  Q b S<br><br>*Loss: 3.1831*|Koould, uve'lle wonnd th wr owham or w I'mer co Thoousessald 6or rouse ayl<br>[Ct eranere-Che An uortest erres])<br>[Shol st yorst loheneuran yighim<br>Aneve I you' c iq coeeou<br>Thewhon ing ct bu y wero y Byv<br><br>*Loss: 2.4845*|
|1500|I've menibu I olds f outhealike be<br>Andes alsont'malssevee<br>Ash, ifeerer n, I' mecepe]<br>I watowhare ig I-deringugengoft e th"Oas<br>Lat it yo (Sait me (He eve hancaaime s’mt and ar ngan wea bet neatst yond<br><br>*Loss: 2.3532*|With your feels mind<br>Ail I catcheres you were words<br>You don't know what I was lup in the scar?<br>Are you just the we stay<br>I never heel the sky?<br>Just brokers for your night<br>Something what'se, love you mi<br><br>*Loss: 1.6168*|
|3000|Whe ar wandy, warit totna ndig<br>The I cesthe I brso thaint at's mit freate of heme head wor oul<br>I merednnd ous and wo deemtho I yoren, she drcemow, an'tin, ghas's<br>I olof wan die bewaun tt'men w ca was<br><br>*Loss: 2.1478*|And you what a mall placion<br>And you know places in into mind-seide<br>And you don't know that<br>Forgo that was red, right<br>I would feel wond[Verse 2]<br>You could see that this magic by this best<br>But the ones<br><br>*Loss: 1.6058*|
|4900|[Verse Thirs] Oh<br>She you wenou lingt hink you nory gray bed<br>ass't ait bacallling you're tought a nof ye[Choler 2]<br>You wald scaut driry, slak aone[Chorus]<br>You you and whant you<br>You'r the be golonat, sa<br><br>*Loss: 1.9927*|Heed for the ontowns in begin<br>Even if you're touching my mind<br>I heard to know his you were that fighting to you<br>You might also like[Pre-Chorus]<br>And we had you so go<br>Trake one who side joked a blue sin<br><br>*Loss: 1.6535*|

<p align="center"><i><b>Table 2</b> | Progression of text generation (200 tokens) at different training steps using two Embedding dimensions on the Taylor Swift dataset</i></p>

It's worth noting that the models generated text of surprisingly good quality, considering they were trained on a relatively small dataset. However, there are still numerous grammatical errors, and the output largely lacks coherent semantics and meaning. The text was clearly generated without much consideration for context, indicating significant room for improvement.

---

## Exercise 2: Working with Real LLMs
As previously seen, my toy GPT can only take me this far. Language Modeling is a field in which models need a real *ton* of text data of any kind and topic in order to generate meaningful text. So, small companies and researchers or students must start using **pre-trained** Language Models.
Luckily, the [HuggingFace](https://huggingface.co/) ecosystem allows to access a *huge* variety of pre-trained Transformer models and datasets for any kind of application. In this exercise, I will learn to use stuff from HuggingFace for my future Language Modeling projects.

### Exercise 2.1: Installation and text tokenization
First things first, I need to install the [HuggingFace Transformer library](https://huggingface.co/docs/transformers/index) on my `conda` environment.

The language model I will focus on is **GPT2**. The key classes that I'll work with are `GPT2Tokenizer`, used to encode text into sub-word tokens, and `GPT2LMHeadModel`, a version of GPT2 that has *text prediction heads* attached to the final hidden layer representations (which is what we need to **generate** text).

While inputting the sequence *'Hello World'*, I noticed that the tokenizer has a single integer for representing the whole "Hello" and "World" words. This suggests that many existing English (and possibly other language) words are encoded as a single integer.

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

It's important to note that the default inference mode for GPT2 is *greedy*, which might not result in satisfying generated text. Yet, the `generate()` function of HuggingFace's GPT2 model offers several parameters to control the quality of generated text.

+ `do_sample` determines whether to use random sampling or greedy search, affecting the balance between creativity and coherence.
+ `temperature` adjusts the randomness of predictions: lower values produce more determined text, while higher values increase creativity but potentially reduce coherence.
+ `early_stopping` can be useful for generating more concise texts by halting generation when all beams have reached EOS (the End Of Sentence).
+ `no_repeat_ngram_size` helps prevent phrase repetition, improving text variety.

Optimizing these parameters requires experimentation, as their effect may vary depending on the specific context and application. In order to qualitatively evaluate the performance of the `generate()` function and the effect of its arguments, I fixed the text prompt to be the same at every run.

Here is some text generated tuning the function arguments (all outputs are available at `2_textgeneration/generation_logs.txt`):

**Prompt**: *"The main goal in life is "*

|*Function Arguments*|*Text Generated*|
|:---:|:---:|
|`do_sample = False`<br>`temperature = 1.0`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 0`|*The main goal in life is* to be able to do something that you love.<br>I'm not saying that I'm going to do anything that I love,<br>but I'm saying that I'm going to do something that I love.<br>I'm not saying that I'm going to do anything that I love,<br>but I'm saying that I'm going to do something that I love.<br>I'm not saying that I'm going to do anything that I love,<br>but I'm saying that I'm going to do something that I love.|
|`do_sample = False`<br>`temperature = 1.5`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 0`|*The main goal in life is* to be able to do something that you love.<br>I'm not saying that I'm going to do anything that I love,<br>but I'm saying that I'm going to do something that I love.<br>I'm not saying that I'm going to do anything that I love,<br>but I'm saying that I'm going to do something that I love.<br>I'm not saying that I'm going to do anything that I love,<br> but I'm saying that I'm going to do something that I love.|
|`do_sample = False`<br>`temperature = 1.0`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 3`|*The main goal in life is* to be able to do something that you love.<br>I'm not saying that I'm going to do anything that I love,<br>but I'm saying that it's important to be a good person.<br>I think that's what makes you a good human being.<br>I think that the best way to be good is to do what you love, and that's to be yourself.<br>I don't think that you can do anything without being yourself.<br>.(Laughter.)<br>.[endoftext]|
|`do_sample = True`<br>`temperature = 1.0`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 0`|*The main goal in life is* to get your car running.<br>When you move around, you want to be there if you need help.<br>You want to stay where you are. What does that mean? Well, if you spend your life riding a little bit,<br>it's going to take on a life of its own. When you want to go on a new adventure,<br>you want to find peace in there, and you try it in every city.<br>In Boston, I had a different path to take with the bus.<br>I went on my first bike trip and it was so fun that I went to another city with that.<br>Those city buses were so good.<br>Those are the buses that came out the door when I was a kid.|
|`do_sample = True`<br>`temperature = 1.5`<br>`early_stopping = False`<br>`no_repeat_ngram_size = 0`|*The main goal in life is* never just to find your passion for the community anymore;<br>everything is based around it. I've only taken this to extremes<br>in terms of spending so much money in the past; I've spent less on my lifestyle this term anyway.<br>I have started my job to become more efficient and do just that - or<br>better known as what I'm calling working without even having much, and if in my first year<br>at the restaurant in September I really needed some help I have been doing this long.<br>My business has become one of the very first businesses outside Toronto<br>and every year at this point has increased steadily.<br>One year I just hit my first financial goals.|
|`do_sample = True`<br>`temperature = 0.3`<br>`early_stopping = True`<br>`no_repeat_ngram_size = 0`|*The main goal in life is* to be happy.<br>"I'm not going to be happy if I'm not happy," he said.<br>"I'm not going to be happy if I'm not happy. I'm not going to be happy if I'm not happy.<br>I'm not going to be happy if I'm not happy."<br>The former NBA player said he was "very happy" to be back in the NBA.<br>"I'm not going to be happy if I'm not happy," he said.<br>"I'm not going to be happy if I'm not happy. I'm not going to be happy if I'm not happy."|

<p align="center"><i><b>Table 3</b> | Comparison of text generated using different parameters of the generate() function in HuggingFace's GPT2 model</i></p>

The default configuration for the `generate()` function results in *greedy* generation, which produces text with little to no sense. The output is typically a simple sentence repeated until the `max_new_tokens` limit is reached.

When `do_sample = False`, the generation is greedy and deterministic. This is comparable to having a `temperature` value close to zero, as sampling (which introduces variability) is disabled in favor of a greedy approach to generation.

Setting `do_sample = True` enables more diverse generation by allowing sampling of varied token sequences, rather than always producing the same greedy text.

As lower temperatures freeze generation, higher `temperature` values lead to hotter, "noisier" and more unpredictable generation. Very high temperatures can result in nonsensical output, including random symbols and non-existent words.

---

## Exercise 3: Reusing Pre-trained LLMs
In each of the following exercises, I'm tasked with adapting a pre-trained Large Language Model (LLM) to a new Natural Language Understanding task.
I've chosen to use `DistilBERT` to gain a better understanding of how **BERT** models function.

**BERT** models (including DistilBERT) have a special [CLS] token prepended to each latent representation output from a self-attention block. I will use this token directly as a representation for addressing my subsequent tasks.

### Exercise 3.1: Training a Text Classifier
An important decision in this process is the selection of an appropriate dataset for the task. After looking at the [text classification datasets on HuggingFace](https://huggingface.co/datasets?task_categories=task_categories:text-classification&sort=downloads), one standout option is [*ag_news*](https://huggingface.co/datasets/fancyzhx/ag_news). This dataset is of moderate size and features multi-class classification of global news, with 4 perfectly balanced categories: *[World, Sports, Business, Sci/Tech]*.

However, I found myself drawn to a more challenging option: the [*dair-ai/emotion*](https://huggingface.co/datasets/dair-ai/emotion) dataset. In contrast to *ag_news*, this dataset focuses on sentiment analysis of text snippets and appears considerably more challenging, because it has limited data and 6 unevenly distributed classes: *[joy, sadness, anger, fear, love, surprise]*.

I decided to use the *DistilBERT* model *exclusively* as a feature extractor, followed by training a shallow model (an MLP, or even a Logistic Regression) to classify the data using an appropriate loss function.

Features for each dataset are extracted using the `feature_extractor` function in the `my_utils.py` script, and stored in the `31_textclassification` folder. The T-SNE visualization of these features reveals how effectively DistilBERT separates the representations of the classes in the *ag-news* dataset, while struggling in the *dair-ai/emotion* dataset.

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab2/images/3_textclassification/ag-news/tsne_train_mlp.png" width="40%">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab2/images/3_textclassification/ag-news/tsne_val_mlp.png" width="40%">
</p>
<p align="center"><i><b>Figure 2</b> | T-SNE visualization of <b>train</b> (left) and <b>test</b> (right) features extracted from the <b>ag_news</b> dataset</i></p><br>

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab2/images/3_textclassification/dair-ai/tsne_train_mlp_10.png" width="40%">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab2/images/3_textclassification/dair-ai/tsne_val_mlp_10.png" width="40%">
</p>
<p align="center"><i><b>Figure 3</b> | T-SNE visualization of <b>train</b> (left) and <b>test</b> (right) features extracted from the <b>dair-ai/emotion</b> dataset</i></p><br>

#### Text Classification results

On the *ag_news* dataset:

| |*MLP*|*Logistic Regression*|
|:-:|:-:|:-:|
|**Accuracy**|91.76%|91.51%|
|**F1 Score**|0.9174|0.9151|
|**Precision**|0.9174|0.9152|
|**Recall**|0.9176|0.9151|

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab2/images/3_textclassification/ag-news/confusion_matrix_mlp.png" width="40%">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab2/images/3_textclassification/ag-news/confusion_matrix_logreg.png" width="40%">
</p>
<p align="center"><i><b>Figure 4</b> | Confusion Matrices for classifications using an <b>MLP</b> trained for 10 epochs (left) and a <b>Logistic Regression</b> (right) on the <b>ag_news</b> dataset</i></p><br>

On the *dair-ai/emotion* dataset:

| |*MLP*|*Logistic Regression*|
|:-:|:-:|:-:|
|**Accuracy**|66.30%|65.25%|
|**F1 Score**|0.6522|0.6437|
|**Precision**|0.6569|0.6416|
|**Recall**|0.6630|0.6525|

<p float="center" align="center">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab2/images/3_textclassification/dair-ai/confusion_matrix_mlp_10.png" width="40%">
  <img src="https://github.com/giovancombo/DeepLearningApps/blob/main/lab2/images/3_textclassification/dair-ai/confusion_matrix_logreg_10.png" width="40%">
</p>
<p align="center"><i><b>Figure 5</b> | Confusion Matrices for classifications using an <b>MLP</b> trained for 10 epochs (left) and a <b>Logistic Regression</b> (right) on the <b>dair-ai/emotion</b> dataset</i></p>

---

### Exercise 3.2: Training a Question Answering Model (WORK IN PROGRESS)
The Question-Answering task consists in predicting and giving the correct answer at particular contextualized multiple-choice questions.
To address it, I will pick one dataset from the [multiple choice question answering datasets on HuggingFace](https://huggingface.co/datasets?task_categories=task_categories:multiple-choice&sort=downloads). Even here, for computation purposes, it is good to choose a *moderately* sized one.
Here, I *might* be able to avoid fine-tuning by training a simple model to *rank* the multiple choices using *margin loss* in PyTorch.

---

### Exercise 3.3: Training a Retrieval Model (WORK IN PROGRESS)
The HuggingFace dataset repository contains a large number of ["text retrieval" problems](https://huggingface.co/datasets?task_categories=task_categories:text-retrieval&p=1&sort=downloads). These tasks generally require that the model measure *similarity* between text in some metric space.
Naively, just a *cosine similarity* between [CLS] tokens could get me pretty far.

I chose the text retrieval problem.

**Tip**: Sometimes identifying the *retrieval* problems in these datasets can be half the challenge. [This dataset](https://huggingface.co/datasets/BeIR/scifact) might be a good starting point.
