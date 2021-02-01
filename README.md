<h1 align="center"><strong>🔎 Extractive Summarizer using KoBERT</strong></h3>

`#KoBERT` `#NLP` `#Node.js` `#Express`

# 🚦 1. Introduction
 - 세줄요약좀! `2020-2학기 YBIGTA 컨퍼런스`
 - KoBERT를 활용하여 Extractive summarizer를 학습시키고 Node.js를 활용하여 웹으로 구현해보았다.
 - 참여자: 김지수, 문승현, 양정열, 유승수, 윤형준, 정현우

---

# 🚦 2. Training
  - KoBERT를 활용하여 학습시키는 도중에 Pytorch가 계속 터져서 몇가지 모듈은 직접 만들어서 학습시켰다.
  - 학습에는 Google Colab Pro TPU를 사용하였고, 데이터는 데이콘의 **한국어 문서 추출요약 AI 경진대회** 데이터를 사용하였다.
  - [Data Source](https://dacon.io/competitions/official/235671/data/)
  - [Colab version](https://colab.research.google.com/github/hw79chopin/3-lines-summary/blob/master/training/KoBERT%20Training%20(TPU).ipynb) GPU가 터져서 TPU로 학습하였다.
  
 
---
 
# 🚦 3. Web 시연하기

## 3-1) 필요 라이브러리 설치 (Node.js)
- 우선 이 github repo를 다운 or git clone 하기
- 그 다음에 [Web] 폴더에 들어가서 cmd창을 실행한다.
- 그리고 밑에 명령을 차례대로 입력한다.
```console
$ npm init
$ npm install --save-dev nodemon
$ npm install --save express body-parser ejs mysql2 sequelize express-session express-session-sequelize request-promise
```

## 3-2) 필요 라이브러리 설치 (Python)
- cmd창을 열고 아래 명령들을 입력한다.
```console
$ pip install --upgrade pip
$ pip install pymysql numpy pandas genism flask
```
- cmd창에서 `env.sh` 파일을 실행시키면 KoBERT를 설치해준다.  

## 3-4) Web 시작하기
- [Web] 폴더에서 cmd창을 열어주고 아래 명령을 입력한다.
```console
$ npm start
```

- [Web] 폴더 내 [python (flask)] 폴더에서 cmd창을 열어주고 아래 명령을 입력한다.
- flask 서버를 통해서 요약 결과를 node 서버와 주고 받아야 해서 flask도 실행해주셔서 합니다.
```console
$ flask run
```

- 크롬을 열어서 주소창에 http://localhost:3000/를 입력하면 세줄요약기를 체험할 수 있다.

---

<h3 align="center"><strong>끗! 🙌</strong></h3>
