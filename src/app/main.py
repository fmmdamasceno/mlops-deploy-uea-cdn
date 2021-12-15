# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import os

from datetime import datetime
# from pubsub import

from flask import Flask, request, jsonify
from transform_data import encode_data
from flask_basicauth import BasicAuth

features = ['gender',
            'enrolled_university',
            'education_level',
            'major_discipline',
            'experience',
            'company_size',
            'company_type',
            'last_new_job',
            'relevent_experience',
            'training_hours']

app = Flask(__name__)
app.config["BASIC_AUTH_USERNAME"] = os.environ.get('BASIC_AUTH_USERNAME')
app.config["BASIC_AUTH_PASSWORD"] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)


def load_model(file_name=''):
    return pickle.load(open(file_name, "rb"))


model = load_model('models/model_v0.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    encoded_data = encode_data(data)

    df_to_predict = pd.DataFrame(encoded_data, index=[0])
    name = df_to_predict['name'][0]
    df_to_predict = df_to_predict.drop(['name'], axis=1)

    result = int(model.predict(df_to_predict)[0])

    status = "Procurando emprego"
    if result == 0:
        status = "Não procurando emprego"

    resultado = jsonify(name=name, status=status)

    return resultado


@app.route("/")
def home():
    return "API de predição"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
