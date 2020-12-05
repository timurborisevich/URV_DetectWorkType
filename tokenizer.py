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