# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab2

# Code for EXERCISE 2

import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from tqdm.auto import tqdm

# Saving the generated text in a txt file?
save_generation = True
generation_pipeline = False

# Hyperparameters for text generation
max_new_tokens = 200
do_sample = True
temperature = 1.2
early_stopping = True
no_repeat_ngram_size = 2

# Input prompt text for generation
input_text = "The main goal in life is"
# input_text = input("What do you want to say?\n")


if __name__ == '__main__':
    # Creating a subword tokenizer from GPT2 pretrained model: new vocab_size = 50,257
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Loading the pretrained model: setting the padding token as the end of sequence token
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id = tokenizer.eos_token_id)

    # Converting string inputs to sequences of tokens: let's compare the input length with the encoded sequence length
    tokenized_text = tokenizer.encode(input_text, return_tensors = 'pt')
    print(f"Input string:\t\t{input_text}\nTokenized string:\t{tokenized_text}")

    # Let's generate some text from the input sequence
    print(f"Input text: {input_text}\nWaiting for the generation of new text...")
    if generation_pipeline:
        # An alternative way for generating text is to use a HuggingFace pipeline
        generator = pipeline('2_text-generation', model = model, tokenizer = tokenizer)
        print(f"Generating through Transformers Pipeline\n\nInput text:\t{input_text}\nWaiting for the generation of new text...\n")
        print("Text generated!\n",
            generator(input_text, max_new_tokens = max_new_tokens,
                                  do_sample = do_sample,
                                  temperature = temperature,
                                  early_stopping = early_stopping,
                                  no_repeat_ngram_size = no_repeat_ngram_size)[0]['generated_text'])
    else:
        tokenized_text = tokenizer(input_text, return_tensors = 'pt')
        generated_text = tokenizer.decode(model.generate(tokenized_text['input_ids'],
                                                         max_new_tokens = max_new_tokens,
                                                         do_sample = do_sample,
                                                         temperature = temperature,
                                                         early_stopping = early_stopping,
                                                         no_repeat_ngram_size = no_repeat_ngram_size)[0].tolist())
        print(f"\nText generated!\n{generated_text}")

    # Saving the generated text in a txt file
    if save_generation:
        folder = "2_textgeneration"
        os.makedirs(folder, exist_ok = True)
        with open(f"{folder}/generation_logs.txt", 'a') as f:
            f.write(f"HPs: max_new_tokens = {max_new_tokens}, do_sample = {do_sample}, temperature = {temperature}, early_stopping = {early_stopping}, no_repeat_ngram_size = {no_repeat_ngram_size}\n\n")
            f.write(f"Input text: {input_text}\nGenerated text: {generated_text}\n\n- - - - - - - - - - - - - - - -\n")
        print(f'\nText saved in \'{folder}/generation_logs.txt\'')
