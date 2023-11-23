#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Set-up create_sumamry

import torch
import operator
from collections import defaultdict
from transformers import AutoTokenizer, BartForConditionalGeneration
from krwordrank.word import KRWordRank
from sentence_transformers import SentenceTransformer


###############################################################################################################################
### 개별 요약문 생성 단계 전체 과정 ###
### create_summary() 함수가 동작 ###
###############################################################################################################################


# 크롤링된 기사의 원문에 남아있는 불필요한 문자열을 제거하기 위한 함수
# Input : '기사 원문 string'
# Output : '특수문자가 제거된 기사 원문'
def remove_substrings(raw_text):
    target_string = "다."
    must_remove_char = [";", "▶", "ⓒ", "©", "☞", "&", "★", "☆", "✩", "&nbsp"]
    substrings = set()
    s = raw_text
    if target_string in s:
        last_occurrence = s.rfind(target_string)
        substring = s[last_occurrence + len(target_string):]
        if substring and any(char in substring for char in must_remove_char):
            substrings.add(substring)
            s = s[:last_occurrence + len(target_string)]  # substring 제거
    return s
# removed_sentence = remove_substrings('string')

###############################################################################################################################

# 학습된 요약 모델/토크나이저를 불러오는 함수
# Input : X
# Output : 학습된 pt를 weight로 사용하는 모델
def get_model():
    # model 폴더 안에 학습된 model.pt 파일 넣기
    model_path = "/srv/ml-server/model/model.pt"
    model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v2')
    return tokenizer

###############################################################################################################################



# 긴 문장 전용 요약 함수
# 1. 기사 내 키워드에 대한 점수 dict 추출 함수
# Input : 'article : 긴 기사 원문 string'
# Output : '해당 기사의 keyword list'
def extract_keyword_scores(article, min_count=2, max_length=3, verbose=False, stopwords=[]):
    # 문장 단위로 나누기
    sentences = article.split(".")

    wordrank_extractor = KRWordRank(
        min_count = min_count,   # 단어의 최소 출현 빈도수
        max_length = max_length, # 단어의 최대 길이
        verbose = verbose
    )

    keywords, rank, graph = wordrank_extractor.extract(sentences)
    for stopword in stopwords:
        keywords.pop(stopword, None)

    return keywords

# 2. 키워드 점수 dict를 이용해 기사 내의 문장들에 대해 중요도 점수를 계산하는 함수
# Input : 'article : 긴 기사 원문 string, keyword_score : 1에서 생성된 키워드 스코어 list'
# Output : '각 문장의 score list'
def compute_sentence_scores(article, keyword_scores):
    sentences = article.split(".")
    sentence_scores = defaultdict(float)

    for sentence in sentences:
        for word, score in keyword_scores.items():
            if word in sentence:
                sentence_scores[sentence] += score

    return sentence_scores


# 3. 가장 낮은 점수를 가진 문장을 제거하면서 기사 전체 길이가 1024 이하가 되도록 만드는 함수
# Input : '긴 기사 원문 string', '각 문장의 score list'
# Output : '짧은 기사 원문 string'
def trim_article(article, sentence_scores, max_length=1024):
    sentences = article.split(".")
    tokenizer = get_tokenizer()

    while len(tokenizer.encode(article)) > max_length:
        # 가장 점수가 낮은 문장 찾기
        min_score_sentence = min(sentence_scores.items(), key=operator.itemgetter(1))[0]
        # 가장 점수가 낮은 문장 제거
        if min_score_sentence in sentences:
            sentences.remove(min_score_sentence)
        # 제거된 문장의 점수 정보도 제거
        sentence_scores.pop(min_score_sentence)

        # 기사 다시 조합
        article = '.'.join(sentences)

    return article

# 4. 긴 문장의 기사의 길이를 줄여주는 함수
# Input : '긴 기사 원문 string'
# Output : '짧은 기사 원문 string'
def summarize_article(article, max_length=1024, min_count=2, max_length_word=3, verbose=False, stopwords=[]):
    # 1. 기사 내 키워드에 대한 점수 dict 추출
    keyword_scores = extract_keyword_scores(article, min_count, max_length_word, verbose, stopwords=stopwords)

    # 2. 키워드 점수 dict를 이용해 기사 내의 문장들에 대해 중요도 점수를 계산
    sentence_scores = compute_sentence_scores(article, keyword_scores)

    # 3. 가장 낮은 점수를 가진 문장을 제거하면서 기사 전체 길이가 1024 이하가 되도록 만드는 과정
    summarized_article = trim_article(article, sentence_scores, max_length)

    return summarized_article



# 크롤링된 기사의 원문을 입력하면 요약문을 반환해주는 함수
# Input : '기사 원문 string', 'keyword list'
# Output : '요약문 string'
def create_summary(raw_text, keywords, model, tokenizer, device):
    #불필요 텍스트 제거
    input_text = remove_substrings(raw_text)
    input_text = input_text.replace('\n', ' ')

    #short article case
    if(input_text.find("[속보]") != -1 and input_text.find("[속보]") <= input_text.find("다.")):
        start = input_text.find("[속보]") + len("[속보]")  # [속보] 다음 위치 계산
        end = input_text.find("\n", start)  # 다음 줄바꿈 위치 계산
        summ = input_text[start:end]  # 추출된 문자열
        return summ

    #short article case
    if(input_text.find("(속보)") != -1 and input_text.find("[속보]") <= input_text.find("다.")):
        start = input_text.find("(속보)") + len("[속보]")  # [속보] 다음 위치 계산
        end = input_text.find("\n", start)  # 다음 줄바꿈 위치 계산
        summ = input_text[start:end]  # 추출된 문자열
        return summ

    #generate summary by model
    keywords_prefix = '(키워드 : '
    for i, kw in enumerate(keywords):
        keywords_prefix += kw
        if i != len(keywords) - 1:
            keywords_prefix += ', '
        else:
            keywords_prefix += ")"

    raw_input_ids = tokenizer.encode(keywords_prefix + input_text)

    # 긴문장 처리
    if(len(raw_input_ids) > 1024):
        shorten_text = summarize_article(input_text)
        raw_input_ids = tokenizer.encode(keywords_prefix + shorten_text)

    input_ids = [tokenizer.bos_token_id] + raw_input_ids + [tokenizer.eos_token_id]

    summary_ids = model.generate(torch.tensor([input_ids]).to(device),
                                max_length=1024,
                                early_stopping=True,
                                repetition_penalty=2.0)
    summ = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)

    if(summ.find("다.") != -1):
        summ = summ.split("다.")[0] + "다."

    return summ


# In[ ]:


# Set-up Extract Keyword

import json
import time
from krwordrank.word import summarize_with_keywords
from konlpy.tag import Kkma
from konlpy.tag import Okt
import numpy as np
from tqdm import tqdm
import re

class SentenceTokenizer(object):
    def __init__(self, stopwords):
        self.kkma = Kkma()
        self.okt = Okt()
        self.stopwords = stopwords

    # sentences 단위로 구분
    def text2sentences(self, text):
        sentences = self.kkma.sentences(text)
        return sentences

    # sentences에서 명사를 추출
    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence != '':
                nouns.append(' '.join([noun for noun in self.okt.nouns(str(sentence))
                  if noun not in self.stopwords and len(noun) > 1]))
        return nouns



def create_stopword_set(stopword_path):
  stopwords_new = set(line.strip() for line in open(stopword_path, "r", encoding="utf-8"))
  stopwords = {'위해', '지원', '이번', '시장', '있다', '서비스', '대한', '밝혔다', '기술', '있는', '위한', '것으로', '말했다', '올해', '사업', '국내', '기업', '있습니다', '20', '지난', '통해'} # 이전에 추출된 키워드 반영
  stopwords.update(stopwords_new)
  return stopwords

def get_keywords_top5(text, stopwords):

  min_count = 2     # 단어의 최소 출현 빈도수 (그래프 생성 시)
  max_length = 3    # 단어의 최대 길이
  beta = 0.99       # PageRank의 decaying factor beta
  max_iter = 15

  St = SentenceTokenizer(stopwords)
  korean_pattern = '[ㄱ-ㅎㅏ-ㅣ가-힣]+'
  is_hangul = bool(re.search(korean_pattern, text))

  if is_hangul is False:
    print("Error: Not written in Korean.")
    return

  sentences = St.text2sentences(text)
  nouns_sentence = St.get_nouns(sentences)

  # 명사 단위로 추출된 text에서 keywords 구분
  raw_keywords_dict = summarize_with_keywords(nouns_sentence,
                                              min_count=min_count,
                                              max_length=max_length,
                                              beta=beta,
                                              max_iter=max_iter,
                                              stopwords=stopwords,
                                              verbose=False)
  keywords_list = list(raw_keywords_dict.keys())
  top_5 = keywords_list[0:5]
  output = ''
  for keyword in top_5:
    output = output + keyword + ' '

  return output


# In[ ]:


# Set-up Stop words & model & tokenizer

stopword_path = "/srv/ml-server/data/stopwords-ko.txt"
stopwords = create_stopword_set(stopword_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/srv/ml-server/model/model.pt"
model = BartForConditionalGeneration.from_pretrained("gogamza/kobart-base-v2")
tokenizer = AutoTokenizer.from_pretrained('gogamza/kobart-base-v2')

model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
model.to(device)
model.eval()

bert_model = SentenceTransformer('jhgan/ko-sbert-sts')
bert_model.to(device)

# In[ ]:
from datetime import datetime
import json
import os

def main():
    input_path = '/srv/ml-server/input/input.json'
    output_path = '/srv/ml-server/result/summary.json/'
    output_data = []

    with open(input_path, 'r', encoding='utf8') as f:
        input_data = json.load(f)
    for report in input_data:
        content = report['content']
        keyword = get_keywords_top5(content, stopwords)
        summary = create_summary(content, keyword.split(), model, tokenizer, device)

        del report['content']
        report['summary'] = summary
        output_data.append(report)
    
    with open(output_path, 'w', encoding='utf8') as f:
        json.dump(output_data, f)


if __name__ == "__main__":
    main()

def gen_with_api(input_text):
    keyword = get_keywords_top5(input_text, stopwords)
    summary = create_summary(input_text, keyword.split(), model, tokenizer, device)
    return summary
