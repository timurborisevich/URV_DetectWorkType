import joblib
import inspect, os.path
# from tokenizer import tokenize_text_simple_regex
# import logging
# logging.basicConfig(filename='error.log',level=logging.DEBUG)

import pymorphy2 # лемматизация
import re

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

# токенайзер, который приводит в нижний регистр и выбирается слова длиной более 2 символов
def tokenize_text_simple_regex(txt, min_token_size=2):
    txt = txt.lower()
    all_tokens = TOKEN_RE.findall(txt)
    all_tokens = lemmatize(all_tokens)
    return [token for token in all_tokens if len(token) >= min_token_size]

# Загрузка моделей
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
sklearn_pipeline_consult = joblib.load(path + "/Models/ModelForConsult.pkl")
sklearn_pipeline_develop = joblib.load(path + "/Models/ModelForDevelop.pkl")

def DetectWorkType (text, department='consult'):
    if department == 'develop':
        return round(sklearn_pipeline_develop.predict_proba([text])[0][1], 2)
    else:
        return round(sklearn_pipeline_consult.predict_proba([text])[0][1], 2)