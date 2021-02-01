README created

2020-2 YBIGTA conference

- KoBERT를 활용한 Extractive summarizer
Deep Learning Text Summarization

# 2020-11-24 Progress
- BERT Extractive Summarizer를 사용해보았음
- How to install
```console
pip3 install bert-extractive-summarizer
pip install spacy==2.1.3
pip install transformers==2.2.2
pip install neuralcoref
python -m spacy download en_core_web_md
```
- 모든 명령을 다 입력해야 한다!


<h1 align="center" style="background-color:#00FEFE"><strong>🔎 텔레그램 봇이 원하는 정보를 알림으로 준다!</strong></h3>

`#python` `#telegram-bot` 


# 🚦 1. 프로젝트 소개
- 펭지뇽이 나에게 부탁한 텔레그램 알림봇
```console
Pain => 채용공고가 올라왔는지 안 올라왔는지 계속 확인하기가 번거로웠음
Solution => 그래서 원하는 조건의 채용공고가 올라올 때마다 핸드폰으로 알람이 오는 봇을 만들어달라고 부탁한 것.
```
- 여기에 내가 평소에 모아놓은 글귀들 중 다시 봐야될 것들을 매일 랜덤하게 보내주는 quote reminder도 추가해봤다! (ver4.0)
- 카카오톡은 API가 복잡하고 텔레그램이 진짜 간단해서 텔레그램으로 결정!

---

# 🚦 2. 과정
## 2-1. 텔레그램 봇 만들기
- 우선은 텔레그램 봇을 설치해준다.
```python
$ pip install python-telegram-bot
```
- PC 버전 텔레그램을 깐 뒤, 친구 검색창에 이렇게 botfather을 검색한다.

<img src="images/Search_botfather.png" width="50%" height="50%">

- 맨 위에 인증마크가 있는 공식 botfather을 선택한 뒤, 시작을 누르면 다양한 명령어가 나온다.

<img src="images/newbot.png" width="50%" height="50%">

- 시작한 뒤에 /newbot을 입력하고 나오는 설명에 따라서 차근차근 사용자이름, 봇의 이름을 설정한다.  

- 다 끝나면 나오는 `TOKEN` 정보를 꼭 저장해둬야 한다.
<br>

- 설정한 봇의 이름을 검색한 뒤 메세지를 보내면 내가 만든 텔레그램 봇에 메세지가 도착한다!
<br>

- 그 뒤 파이썬으로 가서 이렇게 하면 사용자의 id와 누가 봇으로 메세지를 보냈는지 알 수 있다.
```python
import telegram

# bot 조종하기
bot = telegram.Bot(token='TOKEN이 여기에!')

# 사용자랑 id 알아내기
updates = bot.getUpdates()
for i in updates:
  print(i.message.chat.id, i.message.chat.first_name)
```

## 2-2. 크롤링 코드 만들기 (팜리쿠르트)
- 필요한 정보를 크롤링하는 코드를 만들었다.
- 이번 크롤링에서 특별했던 점은 원하는 text data가 iframe에 있는 형식이어서 그거 해결하는데 애먹었다.
- 코드 전체보기 => [Selenium Version](https://github.com/hw79chopin/Telegram-bot/blob/master/crawling%20bot%20ver1.0/crawler/crawler.py)
- 코드 전체보기 => [BS4 Version](https://github.com/hw79chopin/Telegram-bot/blob/master/crawling%20bot%20ver2.0/crawler.py)

## 2-3. 크롤링 변동사항을 텔레그램으로 알림 보내기
- 코드를 간단하게 살펴보면 요로로콤.
```python
while True:   
    result = crawler.crawl_data()
    for new in sorted(new_info):
      region = result[new]['근무지역']
      title = result[new]['제목']
      URL = result[new]['URL']
      bot.send_message(chat_id='원하는 사용자', text="새로운 공고가 나왔어요!!!\n\n* 지역: {}\n* 제목: <{}>\n\n관심있으면 여길 클릭하숑!\n{}".format(region, title, URL))

    time.sleep(600)
```
- 예전 결과와 비교해서 새로 올라온 공지만 알림 보내는 코드는 코드 전문에 나와있음!
- 코드 전체보기 => [bot.py code](https://github.com/hw79chopin/Telegram-bot/blob/master/bot/bot.py)

## 2-4. AWS Lambda에 올려서 항상 코드가 돌아가게 연결하기
- 이걸 이제 차근차근 하는 중 ㅎㅎ
- 지금은 나의 Gram이 24시간 열일하는중 ㅋㅋㅋ
- Linux로 하면 된다던데....

---

# 🚦 3. 한계 및 아쉬운 점
- 가상서버에 올려서 계속 돌아가게 하면 좋을 듯. 계속 에러가 나거나 컴퓨터가 꺼지면 봇이 멈춘다.

<h3 align="center"><strong>끗! 🙌</strong></h3>
