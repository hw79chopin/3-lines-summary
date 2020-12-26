# -*- coding: utf-8 -*- 
import numpy as np

# pytorch
import torch

# KoBERT
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

# etc
import gluonnlp as nlp

# model
from model.encoder import *
from model.classifier import *
from model.bertsum import *

# utils
from utils import *
from utils.data import *


def tokenizing(tokenizer, article, max_sentence_num=128, max_word_num=128, pad=True, pair=False):

    transform = nlp.data.BERTSentenceTransform(tokenizer, max_seq_length=max_word_num, pad=pad, pair=pair)

    numbers_tokenized = np.array([[]])

    for sentence in article:
        # tokenizing
        tokenized_to_num = transform([sentence])
        tokenized_to_num = np.expand_dims(tokenized_to_num[0], axis=0)
        if numbers_tokenized.shape == (1, 0):
            numbers_tokenized = np.concatenate((tokenized_to_num, ), axis = 0)
        else:
            numbers_tokenized = np.concatenate((numbers_tokenized, tokenized_to_num), axis = 0)

    # padding
    if len(numbers_tokenized) < max_sentence_num:
        remaining_adds = max_sentence_num - len(numbers_tokenized)
        padding_sentence = np.ones((remaining_adds, max_word_num),dtype=np.int32)
        numbers_tokenized = np.concatenate((numbers_tokenized, padding_sentence),axis=0)
    else:
        numbers_tokenized = numbers_tokenized[:max_sentence_num]
      
    numbers_tokenized = numbers_tokenized.astype(np.int32)

    return numbers_tokenized


# def summarize():

if __name__ == "__main__":
    device = 'cpu'

    # bert call
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    max_sentence_num = 64
    max_word_num = 64
    num_workers = 5
    batch_size = 64
    learning_rate = 1e-5

    config = Config({
        "n_enc_vocab": len(vocab),
        "n_dec_vocab": len(vocab),
        "n_enc_seq":128,
        "n_layer":6,
        "d_hidn":128,
        "i_pad":1,
        "d_ff":1024,
        "n_head":4,
        "d_head":64,
        "dropout":0.1,
        "layer_norm_epsilon":1e-12
    })

    # model
    encoder = BertEncoder(bertmodel)
    reducer = DimensionReducer(768, 128)
    second = SecondEncoder(config=config, n_layer=1)
    classifier = BinaryClassifier(128)
    model = BERTSummarizer(config, reducer, second, classifier, device)

    # load model params
    save_path = './checkpoints/checkpoint_1.pt'
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    article = "기아자동차의 대표 SUV 쏘렌토가 올해 국내 시장에서 연간 최다 판매량을 기록할 전망이다. 하이브리드 모델 인기가 판매량을 끌어올린 것으로 분석된다. 25일 관련 업계에 따르면 쏘렌토는 올해 1~11월 국내에서 7만6892대가 팔렸다. 올해 월평균 판매량이 6990대인 점을 고려하면 12월 판매량도 6000대가 넘을 것으로 보인다. 이렇게 되면 연간 판매량도 8만 대를 웃돌 것으로 예상된다. 올해 쏘렌토의 누적 계약량은 이달 중순 이미 10만 대를 넘겼다. 2002년 출시한 쏘렌토의 역대 최다 판매량은 2016년 기록한 8만715대였다. 쏘렌토의 돌풍은 차박(차에서 하는 캠핑) 열풍에 SUV가 인기를 끈 데다, 친환경차에 대한 관심이 커지면서 하이브리드 모델 판매가 늘었기 때문으로 풀이된다. 쏘렌토 하이브리드의 복합 연비는 ℓ당 15.3㎞로 소형차 수준의 높은 연비를 갖췄다. 쏘렌토 하이브리드에 적용된 스마트스트림 터보 하이브리드 엔진은 시스템 최고출력 230마력, 최대토크 35.7㎏f·m의 주행 성능을 발휘한다. 쏘렌토는 충돌 안전성을 높인 4세대 플랫폼과 현대자동차그룹 차량 중 처음으로 ‘다중 충돌방지 자동 제동 시스템’이 적용됐다. 다중 추돌방지 자동 제동 시스템은 1차 사고 발생 후 운전자가 차량을 통제하지 못하는 상황이 발생하면 자동으로 차량을 제동시켜 2차 사고를 막아주는 시스템이다. 또 기아차 처음으로 ‘기아 페이’가 포함됐고, 원격 스마트 주차보조 등의 편의사양도 적용됐다. 기아차 관계자는 “4세대 쏘렌토는 국산 중형 SUV 유일의 하이브리드 라인업을 갖췄다”며 “동급 최고 수준의 실내 공간과 최신 편의·안전 사양에 고객들이 많은 관심을 기울인 것으로 보인다”고 말했다."
    article = article.split('. ')

    tokenized = tokenizing(tok, article, max_sentence_num=128, max_word_num=128, pad=True, pair=False)
    tokenized = torch.from_numpy(tokenized).unsqueeze(0)

    new_batch = embedding(encoder, tokenized, torch.tensor([len(article)]))
    output = model(new_batch, torch.tensor([len(article)])).squeeze()[:len(article)]

    sum_ext = []
    for k in range(3):
        max_v = max(output)
        for idx in range(len(output)):
            if output[idx] == max_v:
                output[idx] = -1
                sum_ext.append(idx)
                break

    summary = [article[i] for i in sorted(sum_ext) if i < len(article)]
    
    print(summary)
# return ". ".join(summary)