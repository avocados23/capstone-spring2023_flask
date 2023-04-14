#!/usr/bin/python3
import pandas as pd
import random
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from flask import Flask, jsonify

app = Flask(__name__)

p = None
p2 = None
p3 = None

def load():
    global p, p2, p3

    if p is None:
        with open('f1.json', 'r') as fin:
            p = model_from_json(fin.read())

    if p2 is None:
        with open('f2.json', 'r') as fin:
            p2 = model_from_json(fin.read())

    if p3 is None:
        with open('f3.json', 'r') as fin:
            p3 = model_from_json(fin.read())

    return p, p2, p3


@app.route('/predict')
def predict():
    p, p2, p3 = load()

    minutes = random.randint(1, 30)

    # extends dataframe
    future = p.make_future_dataframe(periods=minutes, freq='min')

    # calculates predicted values
    forecast = p.predict(future)

    # creates forecasted csv file
    #forecast[['ds', 'yhat']].to_csv('forecast1.csv', index=False)


    # extends dataframe
    future2 = p2.make_future_dataframe(periods=minutes, freq='min')

    # calculates predicted values
    forecast2 = p2.predict(future2)

    # creates forecasted csv file
    #forecast2[['ds', 'yhat']].to_csv('forecast12.csv', index=False)


    # extends dataframe
    future3 = p3.make_future_dataframe(periods=minutes, freq='min')

    # calculates predicted values
    forecast3 = p3.predict(future3)

    # creates forecasted csv file
    #forecast3[['ds', 'yhat']].to_csv('forecast14.csv', index=False)

    #print(forecast['yhat'].iloc[-1], forecast2['yhat'].iloc[-1], forecast3['yhat'].iloc[-1])

    v = str(forecast['yhat'].iloc[-1])
    v2 = str(forecast2['yhat'].iloc[-1])
    v3 = str(forecast3['yhat'].iloc[-1])

    # returns last value for each parking garage
    return jsonify(v, v2, v3)


def train():
    # Sign1_full_fitted.csv
    df = pd.read_csv('Sign1_full_fitted.csv')

    # renames column labels to work with Prophet
    df = df.rename(columns={'ts': 'ds', 'y1': 'y'})

    p = Prophet()
    p.fit(df)


    # Sign12_full_fitted.csv
    df2 = pd.read_csv('Sign12_full_fitted.csv')

    # renames column labels to work with Prophet
    df2 = df2.rename(columns={'ts': 'ds', 'y12': 'y'})

    p2 = Prophet()
    p2.fit(df2)


    # Sign14_full_fitted.csv
    df3 = pd.read_csv('Sign14_full_fitted.csv')

    # renames column labels to work with Prophet
    df3 = df3.rename(columns={'ts': 'ds', 'y14': 'y'})

    p3 = Prophet()
    p3.fit(df3)


    with open('f1.json', 'w') as fout:
        fout.write(model_to_json(p))

    with open('f2.json', 'w') as fout:
        fout.write(model_to_json(p2))

    with open('f3.json', 'w') as fout:
        fout.write(model_to_json(p3))


# do once
# called whenever Flask server is loaded before we predict
if __name__ == '__main__':
    train()
    