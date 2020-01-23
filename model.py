import torch.nn as nn
import torch.nn.functional as F

class TextClassification(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc1 = nn.Linear(embed_dim, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        # self.fc.bias.zero_()

    def forward(self, text):
        embedded = self.embedding(text)
        out = self.fc1(embedded)
        return out
