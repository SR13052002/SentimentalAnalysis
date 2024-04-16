from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb


app = Flask(__name__)

model = load_model('sentiment_model_new.h5')
max_review_length = 500

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def submit():
    name = request.form['name']
    movie_name = request.form['movie']
    parameters = request.form['review']
    word_to_index = imdb.get_word_index()
    words = parameters.split()
    encoded = [1]  # Start with [1] to represent the "start" token
    for word in words:
        if word in word_to_index and word_to_index[word] < 10000:
            encoded.append(word_to_index[word] + 3)  # Shift by 3 to accommodate special tokens
        else:
            encoded.append(2)  # Use [2] to represent "unknown" words
    padded = sequence.pad_sequences([encoded], maxlen=max_review_length)

    # Predict the sentiment score
    prediction = model.predict(padded)
    sentiment_prob = prediction[0]
    sentiment_label = 'Positive' if sentiment_prob >= 0.5 else 'Negative'
    # if sentiment_prob >=0.75:
    #     sentiment_label = "Very Good"
    # elif sentiment_prob < 0.75 and sentiment_prob >=0.5:
    #     sentiment_label = 'Average'
    # elif sentiment_prob < 0.5 and sentiment_prob >=0.25:
    #     sentiment_label = 'Bad'
    # else: sentiment_label='Worst'
    # # return render_template('output.html')
    return render_template('output.html', name=name, movie_name=movie_name, sentiment_label=sentiment_label)

    # return f'Accoridng to {name} the review for {movie_name} is {sentiment_label}'


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
