import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from tokenizer import tokenize_text_simple_regex

# Сохранение модели
import joblib

# считывание и разделение данных на тестовую и обучающую выборки
works = pd.read_excel('data/ОВИК.xlsx')
X_train_seria = works['data']
y_train_seria = works['target']
X_train = X_train_seria.to_list()
y_train = y_train_seria.to_list()

MAX_DF = 0.8 # как часто втречается слово в документах (будем отсекать сверху слова, которые есть везде)
MIN_COUNT = 5 # сколько раз минимум встретилось слово
sklearn_pipeline_OVIK = Pipeline((('vect', TfidfVectorizer(tokenizer=tokenize_text_simple_regex,
                                                      max_df=MAX_DF,
                                                      min_df=MIN_COUNT)),
                             ('cls', LogisticRegression())))

sklearn_pipeline_OVIK.fit(X_train, y_train)
joblib.dump(sklearn_pipeline_OVIK, "Models/ModelForConsult.pkl")