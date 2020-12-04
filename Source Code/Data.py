import pandas as pd
from Levenshtein import ratio
import re
from bert_serving.client import BertClient
import numpy as np

data = pd.read_csv('source.csv')

# this function is used to get printable results
def getResults(questions, fn):
    def getResult(q):
        answer, score, prediction = fn(q)
        return [q, prediction, answer, score]
    return pd.DataFrame(list(map(getResult, questions)), columns=["Q", "Prediction", "A", "Score"])

test_data = [
    "test",
    "Hà Tĩnh thuộc vùng",
    "Địa hình của Bình Định",
    "Lào Cao thuộc vùng nào"
]
data

#print(data.head())


def getApproximateAnswer2(q):
    max_score = 0
    answer = ""
    prediction = ""
    for idx, row in data.iterrows():
        score = ratio(row["Question"], q)
        if score >= 0.9: # I'm sure, stop here
            return row["Answer"], score, row["Question"]
        elif score > max_score: # I'm unsure, continue
            max_score = score
            answer = row["Answer"]
            prediction = row["Question"]

    if max_score > 0.3: # treshold is lowered
        return answer, max_score, prediction
    return "Sorry,No Answer.", max_score, prediction


da=getResults(test_data,getApproximateAnswer2)
#da=pd.to_csv('qq.csv')
#print(da['A']
#print(da['B'])