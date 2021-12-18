# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import os

from datetime import datetime

from flask import Flask
from models import RequestBodyModel, ResponseModel
from mappers import map_data
from flask_basicauth import BasicAuth
from flask_pydantic import validate
from pubsub import publish_to_topic
from utils.env import get_env_var
from utils.constants import MODEL_RESULT_1, MODEL_RESULT_0

app = Flask(__name__)
app.config["BASIC_AUTH_USERNAME"] = get_env_var('BASIC_AUTH_USERNAME')
app.config["BASIC_AUTH_PASSWORD"] = get_env_var('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)


def load_model(file_name=''):
    return pickle.load(open(file_name, "rb"))


model = load_model('models/model.pkl')


@app.route('/predict', methods=['POST'])
@validate()
def predict(body: RequestBodyModel):
    mapped_data = map_data(body)
    userId = body.id
    print(mapped_data)
    df_to_predict = pd.DataFrame(mapped_data, index=[0])

    result = int(model.predict(df_to_predict)[0])

    status = MODEL_RESULT_1
    if result == 0:
        status = MODEL_RESULT_0

    request_datetime = datetime.today().strftime(format="%Y-%m-%d %H:%M:%S")

    msg = ('{"id":"%s", "request_datetime":"%s", "result":%d, "status":"%s"}' % (
        userId, request_datetime, result, status))

    print("Message: %s" % msg)

    publish_to_topic(msg)

    return ResponseModel(id=userId, status=status)


@app.route("/")
def home():
    return "API de predição"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
