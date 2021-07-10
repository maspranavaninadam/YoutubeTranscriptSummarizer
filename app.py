from flask import Flask, render_template, redirect, request
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer
import joblib


""" this function takes youtube video id as input and returns the transcript
if it exists otherwise it return None """

def getTranscript(video_id):
    try:
        data = YouTubeTranscriptApi.get_transcript(video_id)
        text_data = ""
        for value in data:
            text_data += value["text"]
        new_data = ""
        for i in text_data:
            if i == "\n" or i == "\t":
                continue
            else:
                new_data += i
        text_data = new_data
        return text_data
    except:
        return None


""" This function takes transcript as input and returns the summarized text """
def getSummarizedText(transcript):
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = joblib.load("model.pkl")
    inputs = tokenizer.encode("summarize: " + transcript, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=400, min_length=80, length_penalty=5, num_beams=2,
                              no_repeat_ngram_size=2, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0])
    return summary



app = Flask(__name__)


""" Initial display page """
@app.route('/')
def index_page():
    return render_template("index.html")


""" Generate route Handles user input and output """
@app.route('/generate', methods = ['POST'])
def air():
    if request.method == 'POST':
        y_link = request.form["y-link"]
        if y_link == "":
            return render_template("index.html", message="please enter a valid link")
        video_id = y_link.split("=")[1]
        transcript = getTranscript(video_id)
        if transcript is None:
            return render_template("index.html", message="transcript disabled or not available")
        else:
            output = getSummarizedText(transcript)

    return render_template("index.html", summary=output)

if __name__ == '__main__':
    app.run(debug=True)
