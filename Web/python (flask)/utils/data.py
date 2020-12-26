import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# pytorch
import torch
from torch.utils.data import Dataset, DataLoader

import gluonnlp as nlp


class ArticleLineDataset(Dataset):
  
    """
    기사 데이터 로드

    dataframe : 기사 데이터
    article_col : 
    """

    def __init__(self, dataframe, article_col, ext_col, tokenizer, max_sentence_num=128, max_word_num=128, pad=True, pair=False):
        transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=max_word_num, pad=pad, pair=pair)    

        # data
        self.articles = []
        self.sentence_length = []
        for article in tqdm(dataframe[article_col]):
            numbers_tokenized = np.array([[]])

            for sentence in article:
                # tokenizing
                tokenized_to_num = transform([sentence])
                tokenized_to_num = np.expand_dims(tokenized_to_num[0], axis=0)
                if numbers_tokenized.shape == (1, 0):
                    numbers_tokenized = np.concatenate((tokenized_to_num, ), axis = 0)
                else:
                    numbers_tokenized = np.concatenate((numbers_tokenized, tokenized_to_num), axis = 0)

            self.sentence_length.append(np.int32(len(numbers_tokenized)))

            # padding
            if len(numbers_tokenized) < max_sentence_num:
                remaining_adds = max_sentence_num - len(numbers_tokenized)
                padding_sentence = np.ones((remaining_adds, max_word_num),dtype=np.int32)
                numbers_tokenized = np.concatenate((numbers_tokenized, padding_sentence),axis=0)
            else:
                numbers_tokenized = numbers_tokenized[:max_sentence_num]
        
            numbers_tokenized = numbers_tokenized.astype(np.int32)
            self.articles.append(numbers_tokenized)

        # label
        self.summary = []
        for extractive in dataframe[ext_col]:
            binary = self.binary_target_processor(extractive, max_sentence_num)
            self.summary.append(binary)

    def binary_target_processor(self, extractive, max_sentence_num):
        binary = np.zeros((max_sentence_num,1),dtype=np.int32)
        for sentence_index in extractive:
            if sentence_index < max_sentence_num:
                binary[sentence_index] = 1
        return binary

    def __getitem__(self, i):
        return (self.articles[i], self.sentence_length[i], self.summary[i])
    
    def __len__(self):
        return len(self.articles)


class MyDataLoader():
    def __init__(self, dataset, batch_size):
        self.current = 0
        self.dataset = [item for item in dataset]
        self.batch_size = batch_size
        if len(dataset) % batch_size == 0:
            self.stop = len(dataset) // batch_size
        else:
            self.stop = len(dataset) // batch_size + 1
        self.batch = self.devide_batch()

    def devide_batch(self):
        import random

        random.shuffle(self.dataset)

        batchs = [ [] for _ in range(self.stop) ]

        for i, dt in enumerate(self.dataset):
            batchs[i % self.stop].append((torch.from_numpy(dt[0]).unsqueeze(0), torch.tensor(dt[1]).unsqueeze(0), torch.from_numpy(dt[2]).unsqueeze(0)))
        
        return batchs

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.stop:    # 현재 숫자가 반복을 끝낼 숫자보다 작을 때
            flag = True
            for article, num, ext in self.batch[self.current]:
                if flag:
                    articles = torch.cat([article, ], dim=0)
                    nums = torch.cat([num, ], dim=0)
                    exts = torch.cat([ext, ], dim=0)
                    flag = False
                else:
                    articles = torch.cat([articles, article], dim=0)
                    nums = torch.cat([nums, num], dim=0)
                    exts = torch.cat([exts, ext], dim=0)

            self.current += 1           # 현재 숫자를 1 증가시킴
            return [articles, nums, exts]
        else:                           # 현재 숫자가 반복을 끝낼 숫자보다 크거나 같을 때
            self.current = 0
            random.shuffle(self.dataset)
            raise StopIteration
