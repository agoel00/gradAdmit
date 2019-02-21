from flask import Flask, request, jsonify
import joblib
import numpy as np 
from flask_cors import CORS 
import pandas as pd 

app = Flask(__name__)
CORS(app)

reg = joblib.load('./data/gradLR.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    gre = request.args.get('gre')
    toefl = request.args.get('toefl')
    univrating = request.args.get('univrating')
    sop = request.args.get('sop')
    lor = request.args.get('lor')
    cgpa = request.args.get('cgpa')
    research = request.args.get('research')
    
    resp = [gre, toefl, univrating, sop, lor, cgpa, research]

    score = pd.DataFrame(resp).T
    return jsonify(
        {
            'chanceOfAdmit': reg.predict(score).tolist()
        }
    )

if __name__ == '__main__':
    app.run(debug=True)