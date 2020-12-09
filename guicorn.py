# from tokenizer import tokenize_text_simple_regex
from Main import tokenize_text_simple_regex

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


from Main import DetectWorkType

def pars_environ(environ):
    params = environ.split('&')
    params_dic = {}
    for param in params:
        cur_par = param.split('=')
        params_dic[cur_par[0]] = cur_par[1]
    return params_dic

def app(environ, start_response):
    start_response('200 OK', [('Content-Type', 'text/plain')])
    environ = environ['QUERY_STRING']
    params_dic = pars_environ(environ)
    answer = DetectWorkType(params_dic['text'], params_dic['text'])
    return [bytes(str(answer), encoding="utf8")]

bind = "0.0.0.0:5000"
