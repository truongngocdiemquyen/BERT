from bert_serving.client import BertClient
import numpy as np
from Data import data,getResults,test_data
from flask import Flask, redirect, url_for, render_template, request
import timeit

class BertAnswer():
    def __init__(self):
        self.bc = BertClient()
        self.q_data = data["Question"].values.tolist()
        self.a_data = data["Answer"].values.tolist()
        self.questions_encoder = np.load("questions.npy")
        self.questions_encoder_len = np.load("questions_len.npy")

    def get(self, q):
        query_vector = self.bc.encode([q])[0]
        score = np.sum((query_vector * self.questions_encoder), axis=1) / (
            self.questions_encoder_len * (np.sum(query_vector * query_vector) ** 0.5)
        )
        top_id = np.argsort(score)[::-1][0]
        if float(score[top_id]) > 0.94:
            return self.a_data[top_id], score[top_id], self.q_data[top_id]
        return "Xin lỗi, Không có câu trả lời.", score[top_id], self.q_data[top_id]

bm = BertAnswer()

def getBertAnswer(q):
    return bm.get(q)

#test_data =input('Input:')
print('start')

#st = input('Thủ đô của Hà Nội?')
#da = getResults(test_data, getBertAnswer)
#print(da['A'])
#print('don')
app = Flask(__name__)
TEMPLATES_AUTO_RELOAD = True

@app.route("/", methods = ["GET","POST"])
def index():
    if request.method == "POST":
        question = request.form['question']
        if len(question) > 0:
            start = timeit.default_timer()
            #response = PA(question)
            #answer = response.answer
            answer = getResults([question], getBertAnswer)
            stop = timeit.default_timer()
            t = stop - start

            #time = 'Thời gian: ' + str(t) + ' giây'
            qes = 'Câu hỏi: '+ question 
            res =  (answer['A']).unique()

            return render_template('index-test.html' ,qes=qes, res=res)
        else:
            return render_template('index-test.html')
    else:
        return render_template('index-test.html')

if __name__ == "__main__":
    app.run(debug=True)