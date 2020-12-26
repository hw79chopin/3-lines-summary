# pytorch
import torch
import torch.nn as nn


class BERTSummarizer(nn.Module):
    def __init__(self, config, reducer, transformer, classifier, device):
        super().__init__()
        self.config = config
        self.reducer = reducer
        self.second_encoder = transformer
        self.classifier = classifier
        self.device = device

    def forward(self, new_batch, num):
    # reduce dimension for ram space
        new_batch = self.reducer(new_batch)

        # second layer  >> [batch, sentence_num, vector]
        cls_mask = torch.tensor([[1 if i < num[b] else 0 for i in range(new_batch[b].size()[0])] for b in range(len(new_batch))])
        new_batch, attn_prob = self.second_encoder(new_batch, cls_mask)

        # classifier  >> [batch, sentence_num, 1]
        new_batch = self.classifier(new_batch)

        return new_batch