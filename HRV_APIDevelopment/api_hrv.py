import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
app = Flask(__name__)
sleepPredictionModel = pickle.load(open('logistic.pkl', 'rb'))
@app.route('/api/model/hrv', methods=['POST'])
def HRVModelPredictorAPI():
    '''
    Application API for SleepPattern Model
    '''
    dataset = request.get_json(force=True)
    print(dataset)
    dataPayload = request.get_json(force=True)
    # print(dataPayload["TST"])
    df = pd.DataFrame({'age': [dataPayload["age"]], 'sex': [dataPayload["sex"]], 'chest': [dataPayload["chest"]], 'resting_blood_pressure': [dataPayload["resting_blood_pressure"]]
                      , 'serum_cholestoral': [dataPayload["serum_cholestoral"]], 'fasting_blood_sugar': [dataPayload["fasting_blood_sugar"]], 'resting_electrocardiographic_results': [dataPayload["resting_electrocardiographic_results"]], 'maximum_heart_rate_achieved': [dataPayload["maximum_heart_rate_achieved"]],'exercise_induced_angina': [dataPayload["exercise_induced_angina"]],'oldpeak': [dataPayload["oldpeak"]],'slope': [dataPayload["slope"]],'number_of_major_vessels': [dataPayload["number_of_major_vessels"]],'thal': [dataPayload["thal"]]});
    print(df)
    prediction = HRVModelPredictorAPI.predict(df)

    print(prediction)
    return jsonify(str(prediction))

if __name__ == '__main__':
    app.run(debug=True)
