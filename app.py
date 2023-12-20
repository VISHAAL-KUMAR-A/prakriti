from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load the model
regmodel = joblib.load('model.joblib')

d = {0: 'vata', 1: 'pitta', 2: 'kappa', 3: 'vata-pitta',
     4: "vata-kapha", 5: 'pitta-kapha', 6: 'vata-pitta-kapha'}


@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    data = np.array([list(data)])
    print(data)
    output = regmodel.predict(data)
    result = int(output[0])  # Convert to a standard Python integer
    print(result)
    return jsonify(d[result])


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
