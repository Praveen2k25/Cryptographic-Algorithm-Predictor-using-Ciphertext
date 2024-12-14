from flask import Flask, render_template, request
import pandas as pd
import joblib
import math

app = Flask(__name__)

model = joblib.load('../cryptographic.pkl')


le = joblib.load('../label.pkl')



def byte_frequency(ciphertext):
    
    ciphertext = ciphertext.replace(" ", "").lower()
    try:
        bytes_data = bytes.fromhex(ciphertext)
    except ValueError:
        raise ValueError("The ciphertext contains invalid hexadecimal characters.")
    freq = [bytes_data.count(i) for i in range(256)]
    return freq


def calculate_entropy(ciphertext):
    
    ciphertext = ciphertext.replace(" ", "").lower()
    try:
        bytes_data = bytes.fromhex(ciphertext)
    except ValueError:
        raise ValueError("The ciphertext contains invalid hexadecimal characters.")
    if len(bytes_data) == 0:
        return 0.0  
    probability_distribution = [bytes_data.count(i) / len(bytes_data) for i in set(bytes_data)]
    entropy = -sum(p * math.log2(p) for p in probability_distribution if p > 0)
    return entropy



def predict_algorithm(ciphertext):
    
    freq_features = byte_frequency(ciphertext)
    entropy_feature = calculate_entropy(ciphertext)

    
    features = freq_features + [entropy_feature]
    features_df = pd.DataFrame([features])

    
    prediction = model.predict(features_df)

   
    algorithm = le.inverse_transform(prediction)
    return algorithm[0]


@app.route('/', methods=['GET', 'POST'])
def index():
    algorithm = None
    error = None
    if request.method == 'POST':
        ciphertext = request.form.get('ciphertext')
        if ciphertext:
            try:
                algorithm = predict_algorithm(ciphertext)
            except ValueError as e:
                error = str(e)
        else:
            error = "Please enter a valid ciphertext."

    return render_template('index.html', algorithm=algorithm, error=error)


if __name__ == '__main__':
    app.run(debug=True)
