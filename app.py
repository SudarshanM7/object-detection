from flask import Flask, redirect, url_for, render_template, request
from model import recommend_products

app = Flask(__name__)
ml_model_path = 'models/ML_Model.sav'
rec_sys_path = 'models/Recommendation_System.sav'
data_df = 'ab'

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/ml-pred/', methods =["GET", "POST"])
def ml_predictions():
  if (request.method == "POST"):
    uname = request.form.get("username")
    top_5 = recommend_products(uname, data_df, ml_model_path, rec_sys_path)
  text = "The top 5 product recommendations for "+uname+" is<br>"+top_5
  return text

if __name__ == '__main__':
  app.run(debug=True)
  