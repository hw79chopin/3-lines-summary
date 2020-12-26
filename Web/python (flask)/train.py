import pandas as pd
import numpy as np
import random
from tqdm import tqdm, tqdm_notebook

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# KoBERT
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

# etc
import gluonnlp as nlp
from transformers import AdamW
from transformers.optimization import WarmupLinearSchedule

# model
from model.encoder import *
from model.classifier import *
from model.bertsum import *

from utils import *
from utils.data import *



def save_checkpoint(epoch, model, optimizer, scheduler, loss_list, PATH):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_list': loss_list
        }, PATH)


if __name__ == '__main__':


    ###### setting device ######
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f"## Device : {device} ##")


    ####### load KoBERT ########
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


    ###### hyper parameter #####
    max_sentence_num = 64
    max_word_num = 64
    num_workers = 5
    batch_size = 32
    learning_rate = 1e-5
    num_epochs = 1000
    embedding_vector_size = 128

    emsize = 128 # 임베딩 차원
    nhid = 128 # nn.TransformerEncoder 에서 피드포워드 네트워크(feedforward network) 모델의 차원
    nlayers = 2 # nn.TransformerEncoder 내부의 nn.TransformerEncoderLayer 개수
    nhead = 2 # 멀티헤드 어텐션(multi-head attention) 모델의 헤드 개수
    dropout = 0.2 # 드랍아웃(dropout) 값

    config = Config({
        "n_enc_vocab": len(vocab),
        "n_dec_vocab": len(vocab),
        "n_enc_seq":embedding_vector_size,
        "n_layer":6,
        "d_hidn":embedding_vector_size,
        "i_pad":1,
        "d_ff":1024,
        "n_head":4,
        "d_head":64,
        "dropout":0.1,
        "layer_norm_epsilon":1e-12
    })


    ######### load data ########
    dataset = pd.read_json("data/train.jsonl", lines=True)
    print("\n## Data Loading ##")
    dataset = ArticleLineDataset(dataset[:100], "article_original", "extractive", tok, max_sentence_num, max_word_num, True, False)

    # train_loader = MyDataLoader(dataset, batch_size=batch_size)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    ######## model init ########
    encoder = BertEncoder(bertmodel)

    reducer = DimensionReducer(768, emsize)
    # second = SecondEncoder(config=config, n_layer=1)
    second = TransformerModel(emsize, nhead, nhid, nlayers, dropout)
    classifier = BinaryClassifier(emsize)

    model = BERTSummarizer(config, reducer, second, classifier, device)

    # bert model freezing
    for name, param in encoder.named_parameters():
        param.requires_grad = False

    # to device
    encoder = encoder.to(device)
    model = model.to(device)
        

    #### logging model info ####
    print("\n## Model Layer Info ##")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
    print("\n## Model Params Num ##")
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"> {pytorch_total_params}")
    print("\n======================")


    ######## optimizer #########
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # init optimizer, loss
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    criterion = nn.BCELoss().to(device)

    ## linear scheduler 초기화
    t_total = len(train_loader) * num_epochs
    warmup_step = int(t_total * warmup_ratio)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_step, t_total=t_total)


    ########## train ###########
    loss_list = []
    print('\n## Start training ##')
    ## checkpoint 불러와서 학습시킬 경우 range 범위 바꿔주기
    for epoch in range(0, num_epochs):

        model.train()

        for i, (batch, num, label) in enumerate(tqdm_notebook(train_loader)):

            optimizer.zero_grad()

            batch, num, label = batch.long().to(device), num.to(device), label.long().to(device)
            
            # bert embedding
            new_batch = embedding(encoder, batch, num)

            #calculate output
            output = model(new_batch, num)
            
            loss = criterion(output.long().reshape(-1,1).to(device), label.reshape(-1,1))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step() 
            scheduler.step()

            ## accuracy 계산
            answer = 0
            total = 0
            for i in range(len(output)):
                pred = output[i].squeeze()
                answ = label[i].squeeze()
                mask = torch.zeros(output[i].squeeze().shape)
                for k in range(3):
                    max_v = max(pred)
                    for idx in range(len(pred)):
                        if pred[idx] == max_v:
                            mask[idx] = 1
                            pred[idx] = -1
                            break

                for idx in range(len(mask)):
                    if answ[idx] == 1:
                        total += 1
                        if mask[idx] == 1:
                            answer += 1
            accuracy = answer / total

            if (i+1) % 1 == 0: ## 상태 출력
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')
                loss_list.append(loss.item())
            
        ## checkpoint 저장
        if (epoch+1) % 10 == 0:
            if not os.path.exists("checkpoints"):
                os.makedirs("checkpoints")
            save_checkpoint(epoch+1, model, optimizer, scheduler, loss_list, f"checkpoints/checkpoint_{str(epoch+1)}.tar")
            print("checkpoint saved!")

