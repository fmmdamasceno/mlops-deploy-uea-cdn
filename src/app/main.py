# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import os

from datetime import datetime
#from pubsub import 

from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth

features = []

app = Flask(__name__)
app.config["BASIC_AUTH_USERNAME"] = os.environ.get('BASIC_AUTH_USERNAME')
app.config["BASIC_AUTH_PASSWORD"] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)

def load_model(file_name = ''):
    return pickle.load(open(file_name, "rb"))

@app.route("/")
def home():
    return "API de predição"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
