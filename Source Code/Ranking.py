from bert_serving.client import BertClient
import numpy as np
from flask import Flask, redirect, url_for, render_template, request
from Data import data

def encode_questions():
    bc = BertClient()
    questions = data["Question"].values.tolist()
    print("Questions count", len(questions))
    print("Start to calculate encoder....")
    questions_encoder = bc.encode(questions)
    np.save("questions", questions_encoder)
    questions_encoder_len = np.sqrt(
        np.sum(questions_encoder * questions_encoder, axis=1)
    )
    np.save("questions_len", questions_encoder_len)
    print("Encoder ready")

encode_questions()