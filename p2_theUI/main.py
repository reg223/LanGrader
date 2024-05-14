# main.py
# Created by Sam (Kuangyi) Hu

# Purpose:  The core backend architecture using flask

# last updated: May 14th, 2024
# current state: just started

from flask import Flask,render_template
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def hw():
  return render_template('index.html')

