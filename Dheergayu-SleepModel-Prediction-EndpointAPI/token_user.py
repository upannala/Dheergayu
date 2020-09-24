import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)
sleepPredictionModel = pickle.load(open('./pkl_file/randomClassificationModel2020-04-05 19:17:46.136497.pkl', 'rb'))

@app.route('/api/models/hrv', methods=['POST'])
def sleepModelPredictorAPI():
    '''
    Application API for SleepPattern Model
    '''
    dataset = request.get_json(force=True)
    print(dataset)
    dataPayload = request.get_json(force=True)
    # print(dataPayload["TST"])

    df = pd.DataFrame({'TST': [dataPayload["TST"]], 'TIB': [dataPayload["TIB"]], 'SE': [dataPayload["SE"]], 'REM_Density': [dataPayload["REM_Density"]]
                      , 'W': [dataPayload["W"]], 'S1': [dataPayload["S1"]], 'S2': [dataPayload["S2"]], 'S3': [dataPayload["S3"]],'REM': [dataPayload["REM"]]})
    print(df)
    prediction = sleepPredictionModel.predict(df)

# 	"TST":"971",
# 	"TIB":"1024",
# 	"SE":"0.651132",
# 	"REM_Density":"0.230614",
# 	"W":"0.230614",
# 	"S1":"0.230614",
# 	"S2":"0.230614",
# 	"S3":"0.230614",
# 	"Insomnia":"1"
# })])
    print(prediction)
    return jsonify(str(prediction))


if __name__ == '__main__':
    app.run(debug=True)
