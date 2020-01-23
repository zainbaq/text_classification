
import torch
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import text_classification
import numpy as np

NGRAMS = 2

train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None
)

labels = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

MODEL_PATH = 'saved_models/linear_nn.pth'
model = torch.load(MODEL_PATH)

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token] 
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

test_input = "DC United first made contact with Ozil in August as they chased a marque name to replace Derby bound Wayne Rooney. \
    They held more talks with his representatives during a visit to London this week to meet players and agents. \
    The 31-year-old still has 18 months left on his £350,000-a-week and DC United feel they have a better chance of signing him in the winter of 2021."

vocab = train_dataset.get_vocab()
prediction = predict(test_input, model, vocab, NGRAMS)

print(test_input)
print(output)