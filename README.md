# Sentimental_Analysis_Primary_Task
Model The provided code contains a trained model for sentiment analysis using the IMDb movie review dataset. The model is built using Keras with TensorFlow backend. The model architecture consists of an embedding layer, followed by an LSTM layer with 100 units, dropout layer with a rate of 0.5, and a dense layer with a sigmoid activation function. The model is trained for 5 epochs using binary cross-entropy loss and Adam optimizer.

API The provided code contains an API that takes input from an HTML form, where the user can input their name, movie name, and review about that movie. The API predicts the sentiment of the review and displays the output in the output.html file.

To use the API, run the Flask app and go to the home page at http://localhost:5000/. Fill in the form with the required inputs and submit it. The API then preprocesses the input, predicts the sentiment, and displays the result in home.html.

The API loads the trained model from the saved sentiment_model_new.h5 file, which is used to predict the sentiment score. The API preprocesses the input text by encoding the words using the get_word_index() method of the IMDb dataset and padding the encoded sequence to a fixed length of 500 using the pad_sequences() method from Keras. The sentiment score is then predicted using the loaded model's predict() method. The API also assigns a sentiment label to the predicted score, which is displayed in the output file. The sentiment label can be either positive or negarive based on the sentiment score range.
