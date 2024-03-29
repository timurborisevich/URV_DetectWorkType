from urllib.parse import unquote
from tokenizer import tokenize_text_simple_regex
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
    environ = unquote(environ['QUERY_STRING'])
    params_dic = pars_environ(environ)
    answer = DetectWorkType(params_dic['text'], params_dic['department'])
    return [bytes(str(answer), encoding="utf8")]

bind = "0.0.0.0:5555"