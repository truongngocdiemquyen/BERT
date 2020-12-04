from bert_serving.client import BertClient
import numpy as np

def encode_questions2():
    bc = BertClient()
    questions = data["Question"].values.tolist()
    print("Questions count", len(questions))
    print("Start to calculate encoder....")
    questions_encoder = bc.encode(questions)
    np.save("questions2", questions_encoder)
    questions_encoder_len = np.sqrt(
        np.sum(questions_encoder * questions_encoder, axis=1)
    )
    np.save("questions_len2", questions_encoder_len)
    print("Encoder ready")

encode_questions2()

class TunedBertAnswer():
    def __init__(self):
        self.bc = BertClient()
        self.q_data = data["Question"].values.tolist()
        self.a_data = data["Answer"].values.tolist()
        self.questions_encoder = np.load("questions2.npy")
        self.questions_encoder_len = np.load("questions_len2.npy")

    def get(self, q):
        query_vector = self.bc.encode([q])[0]
        score = np.sum((query_vector * self.questions_encoder), axis=1) / (
            self.questions_encoder_len * (np.sum(query_vector * query_vector) ** 0.5)
        )
        top_id = np.argsort(score)[::-1][0]
        if float(score[top_id]) > 0.94:
            return self.a_data[top_id], score[top_id], self.q_data[top_id]
        return "Xin lỗi, Không có câu trả lời.", score[top_id], self.q_data[top_id]

bm2 = TunedBertAnswer()

def getTunedBertAnswer(q):
    return bm2.get(q)

getResults(test_data, getTunedBertAnswer)