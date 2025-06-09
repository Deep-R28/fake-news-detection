from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the trained model and tokenizer
model = tf.keras.models.load_model("fake_news_model.keras")

with open("tokenizer1.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Constants used during training
max_length = 54
padding_type = 'post'
trunc_type = 'post'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    seq = tokenizer.texts_to_sequences([news])
    padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    prediction = model.predict(padded, verbose=0)[0][0]

    result = "This news is ✅ TRUE" if prediction >= 0.5 else "This news is ❌ FAKE"
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)