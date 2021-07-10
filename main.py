import torch
from transformers import AutoTokenizer, AutoModelWithLMHead
import joblib

""" initialising tokenizer to t5-base """
tokenizer = AutoTokenizer.from_pretrained('t5-base')


""" initiating the model """
model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)


""" saving the model into pickle file """
joblib.dump(model, "model.pkl")