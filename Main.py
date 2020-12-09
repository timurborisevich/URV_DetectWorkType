import joblib
import inspect, os.path
from tokenizer import tokenize_text_simple_regex
import logging
logging.basicConfig(filename='error.log',level=logging.DEBUG)

# Загрузка моделей
filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))
print(path)
sklearn_pipeline_consult = joblib.load(path + "/Models/ModelForConsult.pkl")
sklearn_pipeline_develop = joblib.load(path + "/Models/ModelForDevelop.pkl")

def DetectWorkType (text, department='consult'):
    if department == 'develop':
        return round(sklearn_pipeline_develop.predict_proba([text])[0][1], 2)
    else:
        return round(sklearn_pipeline_consult.predict_proba([text])[0][1], 2)