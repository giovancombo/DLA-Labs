# Imports and dependencies
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm.auto import tqdm

# Device configuration
device = torch.device('cpu')

if __name__ == '__main__':

    # Custom input text to be tokenized
    input_text = "Hello World"

    # Creating a subword tokenizer from GPT2 pretrained model: new vocab_size = 50,257!
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Converting string inputs to sequences of tokens: let's compare the input length with the encoded sequence length
    # It's curious to see the difference between calling tokenizer itself, or its encode/tokenize attributes
    tokenized_text = tokenizer.encode(input_text, return_tensors = 'pt')

    print(f"Input string:\t\t{input_text}\nTokenized string:\t{tokenized_text}")


    # Saving configuration
    save_generation = False
    folder = '2_textgeneration'

    # We can even ask the user to input a text prompt
    input_text = "The main goal in life is"         # input("What do you want to say?\n")

    # Hyperparameters for text generation
    max_new_tokens = 100
    do_sample = True
    temperature = 1.2
    early_stopping = True
    no_repeat_ngram_size = 2

    # Loading the pretrained model: setting the padding token as the end of sequence token
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id = tokenizer.eos_token_id)

    print(f"{model.__class__.__name__} instantiated!\n")
    print(f"Input text: {input_text}\nWaiting for the generation of new text...")

    # inputs is a Pytorch tensor
    tokenized_text = tokenizer(input_text, return_tensors = 'pt')

    # Let's generate some text from the input sequence: new_text is a Pytorch tensor
    generated_text = tokenizer.decode(model.generate(tokenized_text['input_ids'],
                                                    max_new_tokens = max_new_tokens,
                                                    do_sample = do_sample,
                                                    temperature = temperature,
                                                    early_stopping = early_stopping,
                                                    no_repeat_ngram_size = no_repeat_ngram_size)[0].tolist())

    print("\nText generated!")
    print(f"{generated_text}")

    # Saving the generated text in a txt file
    if save_generation:
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(f"{folder}/generation_log.txt", 'a') as f:
            f.write(f"HPs: max_new_tokens = {max_new_tokens}, do_sample = {do_sample}, temperature = {temperature}, early_stopping = {early_stopping}, no_repeat_ngram_size = {no_repeat_ngram_size}\n\n")
            f.write(f"Input text: {input_text}\nGenerated text: {generated_text}\n\n- - - - - - - - - - - - - - - -\n")

        print('\nText generated saved in \'{folder}/generation_log.txt\'')


    from transformers import pipeline

    # An alternative way for generating text is to use a Hugging Face pipeline
    generator = pipeline('2_text-generation', model = model, tokenizer = tokenizer)

    print(f"Generating through the pipeline\n\nInput text:\t{input_text}\nWaiting for the generation of new text...\n")
    print("Text generated!\n",
        generator(input_text,
                max_new_tokens = max_new_tokens,
                do_sample = do_sample,
                temperature = temperature,
                early_stopping = early_stopping,
                no_repeat_ngram_size = no_repeat_ngram_size)[0]['generated_text'])