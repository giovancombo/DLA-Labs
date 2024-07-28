# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab2

import os
import torch
import numpy as np
from tqdm import tqdm


# Extracts features and labels from the last hidden layer of a model, or loads them from a folder
def feature_extractor(data, model, tokenizer, batch_size, folder, extract = False, save = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if extract:
        train_features = []
        train_labels = []
        test_features = []
        test_labels = []

        # Extracting features from the Train dataset
        for i in tqdm(range(0, len(data['train']), batch_size), desc = 'Extracting Train Features'):
            encoded_inputs = tokenizer(data['train']['text'][i : i + batch_size], padding = True, truncation = True, return_tensors = 'pt').to(device)
            with torch.no_grad():
                outputs = model(**encoded_inputs)
            encoded_features = outputs.last_hidden_state.mean(dim = 1).detach().cpu().numpy()
            train_features.append(encoded_features)
            train_labels.append(data['train']['label'][i : i + batch_size])

        # Extracting features from the Test dataset
        for i in tqdm(range(0, len(data['test']), batch_size), desc = 'Extracting Test Features'):
            encoded_inputs = tokenizer(data['test']['text'][i : i + batch_size], padding = True, truncation = True, return_tensors = 'pt').to(device)
            with torch.no_grad():
                outputs = model(**encoded_inputs)
            encoded_features = outputs.last_hidden_state.mean(dim = 1).detach().cpu().numpy()
            test_features.append(encoded_features)
            test_labels.append(data['test']['label'][i : i + batch_size])

        # Concatenating the lists of features and labels into a single NumPy array
        # train_features is now a (16000, 768) array: 768 is the embedding dimension
        train_features = np.concatenate(train_features, axis = 0)
        train_labels = np.concatenate(train_labels, axis = 0)
        test_features = np.concatenate(test_features, axis = 0)
        test_labels = np.concatenate(test_labels, axis = 0)

        if save:
            print(f"\nSaving features and labels in \'{folder}\'...")
            os.makedirs(folder, exist_ok = True)
            np.save(f'{folder}/train_features.npy', train_features)
            np.save(f'{folder}/test_features.npy', test_features)
            np.save(f'{folder}/train_labels.npy', train_labels)
            np.save(f'{folder}/test_labels.npy', test_labels)
            print('Features and Labels saved!')
    else:
        print(f"\nLoading features and labels from \'{folder}\'...")
        train_features = np.load(f'{folder}/train_features.npy')
        test_features = np.load(f'{folder}/test_features.npy')
        train_labels = np.load(f'{folder}/train_labels.npy')
        test_labels = np.load(f'{folder}/test_labels.npy')

    return train_features, test_features, train_labels, test_labels
