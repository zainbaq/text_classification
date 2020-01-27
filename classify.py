# Importing libraries
import torch
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import text_classification
import numpy as np

NGRAMS = 2 # change to 3 is three ngrams are used for training
# Update so that vocab download is not necessary
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None
)

# Predict method
def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token] for token in ngrams_iterator(tokenizer(text), ngrams)])
        out = model(text, torch.tensor([0]))
    return out.argmax(1).item() + 1

labels = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

MODEL_PATH = 'saved_models/linear_nn.pth'   # path to saved models
model = torch.load(MODEL_PATH)
model.to('cpu')                             # move model to cpu



test_input = input("CLASSIFY>>", "")        # request input to classify
                                            # from user 

vocab = train_dataset.get_vocab()           # collect vocab - unnecessary if fixed
prediction = predict(test_input, model, vocab, NGRAMS)
print("\nclass: {}".format(labels[prediction]))