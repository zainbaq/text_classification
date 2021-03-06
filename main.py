import torch
import torchtext
from torchtext.datasets import text_classification

from torch.utils.data import DataLoader
NGRAMS = 2 # try 3
import os

import numpy as np

MODEL_PATH = './saved_models'
if not os.path.isdir(MODEL_PATH):
    os.mkdir(MODEL_PATH)

if not os.path.isdir('./.data'):
    os.mkdir('./.data')

train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None
)
BATCH_SIZE = 1 # Only works with BATCH_SIZE = 1 for now
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# vocab = np.save('vocab.npy', train_dataset.get_vocab())

VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUM_CLASSES = len(train_dataset.get_labels())
model = TextClassification(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES).to(device)

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
#     text = text.unsqueeze(0)
    return text, offsets, label

def train_func(sub_train_):

    # Training Model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True, \
    collate_fn=generate_batch)

    for i, (text, offsets, label) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, label = text.to(device), offsets.to(device), label.to(device)

        output = model(text, offsets)

        loss = criterion(output, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == label).sum().item()
    
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, label in data:
        text, offsets, label = text.to(device), offsets.to(device), label.to(device)

        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, label)
            loss += loss.item()
            acc += (output.argmax(1) == label).sum().item()
        
    return loss / len(data_), acc / len(data_)

# Split Dataset and Run Model
import time
from torch.utils.data.dataset import random_split
N_EPOCHS = 10
min_valid_loss = float('inf')

train_losses = []
val_losses = []

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])
print("Training on {}".format(device))

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid)
    train_losses.append(train_loss)
    val_losses.append(valid_loss)
    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

torch.save(model, "saved_models/linear_nn.pth")
print("Optimal model saved.")