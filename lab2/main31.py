# Deep Learning Applications 2023 course, held by Professor Andrew David Bagdanov - University of Florence, Italy
# Created by Giovanni Colombo - Mat. 7092745
# Dedicated Repository on GitHub at https://github.com/giovancombo/DLA_Labs/tree/main/lab2

# Code for EXERCISE 3.1

import os
import torch
import torch.nn as nn
from torch.optim import Adam
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import itertools

from my_utils import feature_extractor
from mlp import MLP, training_mlp


# Generalizing code for handling different datasets and models
datasets = ['ag_news', 'dair-ai/emotion']
name = datasets[1]

pretrained_model = 'distilbert-base-uncased'

# Features Extraction from the dataset
extract_features = False        # If True, extracts the features from the dataset and saves them again (it takes some time)
batch_size = 64

mlp = True
weighted = False                # Weighted Loss for handling class imbalance (for MLP on dair-ai/emotion)

# HPs for training
hidden_size = 512
epochs = 10
learning_rate = 1e-3
logreg_max_iter = 1000

# Saving configuration
save_classification = True
folder = f'31_textclassification/{name}'
os.makedirs(folder + '/images', exist_ok=True)

labels = ["World", "Sports", "Business", "Science/Tech"] if name == datasets[0] else ["sadness", "joy", "love", "anger", "fear", "surprise"]
classes = len(labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_tsne(features, labels, title, filename, metrics):
    tsne = TSNE(n_components=2, random_state=1492).fit_transform(features[:1000])
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne[:, 0], tsne[:, 1], c=labels[:1000], cmap="viridis")
    plt.title(title)
    plt.colorbar(scatter)

    metrics_text = f"Accuracy: {metrics['accuracy']:.2f}%\nF1 Score: {metrics['f1']:.4f}\nPrecision: {metrics['precision']:.4f}\nRecall: {metrics['recall']:.4f}"
    plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(filename)

def plot_confusion_matrix(true_labels, pred_labels, class_labels, title, filename):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)
    fmt = "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(filename)

def evaluate_model(model, X_test, y_test, device):
    if isinstance(model, LogisticRegression):
        pred = model.predict(X_test)
    else:  # Assuming it's an MLP
        model.eval()
        with torch.no_grad():
            outputs = model(torch.tensor(X_test).float().to(device))
            _, pred = torch.max(outputs, 1)
            pred = pred.cpu().numpy()

    accuracy = accuracy_score(y_test, pred) * 100
    f1 = f1_score(y_test, pred, average='weighted')
    precision = precision_score(y_test, pred, average='weighted')
    recall = recall_score(y_test, pred, average='weighted')

    print(f"Prediction completed! Test Accuracy = {accuracy:.2f}%")
    print(f"F1 Score = {f1:.4f}")
    print(f"Precision = {precision:.4f}")
    print(f"Recall = {recall:.4f}")

    metrics = {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}

    return pred, metrics


if __name__ == '__main__':
    # 1) Loading the selected dataset using the load_dataset function from HuggingFace datasets
    data = load_dataset(name)
    print(f"Dataset \'{name}\' loaded!")

    # 2) Instantiating the Tokenizer and the pretrained model
    tokenizer = DistilBertTokenizer.from_pretrained(pretrained_model)
    model = DistilBertModel.from_pretrained(pretrained_model).to(device)

    # Loading features and labels from the folder
    train_features, test_features, train_labels, test_labels = feature_extractor(data, model, tokenizer, batch_size, folder, extract = extract_features, save = save_classification)

    # Training with MLP or Logistic Regression
    if mlp:
        m = "mlp"
        shallow_clf = MLP(hidden_size, classes).to(device)
        optimizer = Adam(shallow_clf.parameters(), lr=learning_rate)
        weights = torch.tensor([3.43, 2.98, 12.27, 7.41, 8.26, 27.97]).to(device) if weighted else None
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Training an MLP...")
        training_mlp(shallow_clf, optimizer, criterion, epochs, batch_size, train_features, train_labels, test_features, test_labels, device)
        pred, metrics = evaluate_model(shallow_clf, test_features, test_labels, device)
    else:
        m = "logreg"
        print("Training a Logistic Regression model...")
        logreg = LogisticRegression(max_iter=logreg_max_iter).fit(train_features, train_labels)
        print("Model trained! Making predictions on new data...\n")
        pred, metrics = evaluate_model(logreg, test_features, test_labels, device)
    
    # Plotting TSNE for both train and test sets, and Confusion Matrix
    plot_tsne(train_features, train_labels, f"TSNE plot of train features for {name} Classification", f'{folder}/images/tsne_train_{m}_{epochs}_w.png', metrics)
    plot_tsne(test_features, test_labels, f"TSNE plot of test features for {name} Classification", f'{folder}/images/tsne_val_{m}_{epochs}_w.png', metrics)
    plot_confusion_matrix(test_labels, pred, labels, f"Confusion Matrix for {name} Classification", f'{folder}/images/confusion_matrix_{m}_{epochs}_w.png')
