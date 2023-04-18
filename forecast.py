#!/usr/bin/python3
import pandas as pd
import requests
import re
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from flask import Flask, jsonify, request

app = Flask(__name__)

p = None
p2 = None
p3 = None

def load():
    print("LOAD")
    global p, p2, p3

    if p is None:
        print("LOAD1")
        with open('f1.json', 'r') as fin:
            p = model_from_json(fin.read())

    if p2 is None:
        print("LOAD2")
        with open('f2.json', 'r') as fin:
            p2 = model_from_json(fin.read())

    if p3 is None:
        print("LOAD3")
        with open('f3.json', 'r') as fin:
            p3 = model_from_json(fin.read())

    return p, p2, p3

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
    if p is None and p2 is None and p3 is None:
        train()
    

@app.route("/")
def index():
    return "Hello World!"

@app.route('/predict')
def predict():
    args = request.args
    if (args.get('latitude') == None) or (args.get('longitude') == None):
        return "Missing latitude or longitude query parameters", 400
    
    latitude = args.get('latitude')
    longitude = args.get('longitude')

    lat_regex_search = re.search('^(\+|-)?(?:90(?:(?:\.0{1,20})?)|(?:[0-9]|[1-8][0-9])(?:(?:\.[0-9]{1,20})?))$', latitude)
    long_regex_search = re.search('^(\+|-)?(?:180(?:(?:\.0{1,20})?)|(?:[0-9]|[1-9][0-9]|1[0-7][0-9])(?:(?:\.[0-9]{1,20})?))$', longitude)

    if not lat_regex_search:
        return "Invalid latitude value", 400

    if not long_regex_search:
        return "Invalid longitude value", 400

    r = requests.get('https://smartparking-backend.herokuapp.com/getMinutes?latitude=' + latitude + '&longitude=' + longitude)
    minutes = r.json()['minutes']

    p, p2, p3 = load()

    # extends dataframe
    future = p.make_future_dataframe(periods=int(minutes), freq='min')

    # calculates predicted values
    forecast = p.predict(future)

    # extends dataframe
    future2 = p2.make_future_dataframe(periods=int(minutes), freq='min')

    # calculates predicted values
    forecast2 = p2.predict(future2)

    # extends dataframe
    future3 = p3.make_future_dataframe(periods=int(minutes), freq='min')

    # calculates predicted values
    forecast3 = p3.predict(future3)

    v = str(forecast['yhat'].iloc[-1])
    v2 = str(forecast2['yhat'].iloc[-1])
    v3 = str(forecast3['yhat'].iloc[-1])

    # returns last value for each parking garage
    return jsonify(v, v2, v3)