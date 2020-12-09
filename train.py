import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import pymorphy2 # лемматизация
import re

# Сохранение модели
import joblib

# считывание и разделение данных на тестовую и обучающую выборки
works = pd.read_excel('data/ОВИК.xlsx')
X_train_seria = works['data']
y_train_seria = works['target']
X_train = X_train_seria.to_list()
y_train = y_train_seria.to_list()

# Лемматизация (приводим слово к канонической форме (лемме))
morph = pymorphy2.MorphAnalyzer()

def lemmatize(text):
    res = list()
    for word in text:
        p = morph.parse(word)[0]
        res.append(p.normal_form)
    return res

# регулярное выражение (1 и более буква или цифра)
TOKEN_RE = re.compile(r'[\w\d]+')

# токенайзер, который приводит в нижний регистр и выбирается слова длиной более 4 символов
def tokenize_text_simple_regex(txt, min_token_size=2):
    txt = txt.lower()
    all_tokens = TOKEN_RE.findall(txt)
    all_tokens = lemmatize(all_tokens)
    # all_tokens = stemming(all_tokens)
    return [token for token in all_tokens if len(token) >= min_token_size]

MAX_DF = 0.8 # как часто втречается слово в документах (будем отсекать сверху слова, которые есть везде)
MIN_COUNT = 5 # сколько раз минимум встретилось слово
sklearn_pipeline_OVIK = Pipeline((('vect', TfidfVectorizer(tokenizer=tokenize_text_simple_regex,
                                                      max_df=MAX_DF,
                                                      min_df=MIN_COUNT)),
                             ('cls', LogisticRegression())))

sklearn_pipeline_OVIK.fit(X_train, y_train)
joblib.dump(sklearn_pipeline_OVIK, "Models/ModelForConsult.pkl")