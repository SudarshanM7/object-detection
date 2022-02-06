from flask import Flask, redirect, url_for, render_template, request
import pandas as pd
import pickle
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from model import predict_top5

# Ignore the warnings
import warnings
warnings.filterwarnings(action='ignore')

app = Flask(__name__)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/ml-pred/', methods =["GET", "POST"])
def ml_predictions():
  if (request.method == "POST"):
    uname = request.form.get("username")
    top_5 = predict_top5()
    text = "The top 5 product recommendations for the username <b>"+uname+"</b> are: <br><br>"+"<br>".join(top_5)
    return text
  elif(request.method == "GET"):
    return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)
