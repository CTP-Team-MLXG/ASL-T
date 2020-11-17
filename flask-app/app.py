import flask
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow

import cv2
from cv2 import cv2

import skimage
import skimage.io
import skimage.transform

app = flask.Flask(__name__, template_folder='templates')

path_to_vectorizer = 'models/vectorizer.pkl'
path_to_text_classifier = 'models/text-classifier.pkl'
path_to_image_classifier = 'models/image-classifier.pkl'

path_to_asl_categories = 'models/CATEGORIES.pickle'

path_to_asl_classifier = 'models/cnn1'

try:
    asl_cnn_classifier = tensorflow.keras.models.load_model(path_to_asl_classifier)
except EOFError as e:
    print(e)

with open(path_to_asl_categories, 'rb') as f:
    ASL_CATEGORIES = pickle.load(f)

with open(path_to_vectorizer, 'rb') as f:
    vectorizer = pickle.load(f)

with open(path_to_text_classifier, 'rb') as f:
    model = pickle.load(f)

with open(path_to_image_classifier, 'rb') as f:
    image_classifier = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':
        # Get file object from user input.
        file = flask.request.files['file']
        
        if file:
            # Read image file string data
            filestr = file.read()
            # Convert string data to np arr
            npimg = np.frombuffer(filestr, np.uint8)
            # Convert np arr to image
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            # Resize the image to match the input the model will accept
            img = cv2.resize(img, (64, 64))
            # Reshape the image into shape (1, 64, 64, 3)
            img = np.asarray([img])

            # Get prediction of image from classifier
            # predictions = asl_cnn_classifier.predict([img])
            prediction = np.argmax(asl_cnn_classifier.predict(img), axis=-1)

            # pred_proba = asl_cnn_classifier.predict_proba(img)
            # print(pred_proba.round(2))

            # Get the value at index of CATEGORIES
            prediction = ASL_CATEGORIES[prediction[0]]

            return flask.render_template('main.html', prediction=prediction)

    return(flask.render_template('main.html'))


    # if flask.request.method == 'POST':
        # # Get the input from the user.
        # user_input_text = flask.request.form['user_input_text']
        
        # # Turn the text into numbers using our vectorizer
        # X = vectorizer.transform([user_input_text])
        
        # # Make a prediction 
        # predictions = model.predict(X)
        
        # # Get the first and only value of the prediction.
        # prediction = predictions[0]

        # # Get the predicted probabs
        # predicted_probas = model.predict_proba(X)

        # # Get the value of the first, and only, predicted proba.
        # predicted_proba = predicted_probas[0]

        # # The first element in the predicted probabs is % democrat
        # precent_democrat = predicted_proba[0]

        # # The second elemnt in predicted probas is % republican
        # precent_republican = predicted_proba[1]


        # return flask.render_template('main.html', 
        #     input_text=user_input_text,
        #     result=prediction,
        #     precent_democrat=precent_democrat,
        #     precent_republican=precent_republican)




@app.route('/input_values/', methods=['GET', 'POST'])
def input_values():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('input_values.html'))

    if flask.request.method == 'POST':
        # Get the input from the user.
        var_one = flask.request.form['input_variable_one']
        var_two = flask.request.form['another-input-variable']
        var_three = flask.request.form['third-input-variable']

        list_of_inputs = [var_one, var_two, var_three]

        return(flask.render_template('input_values.html', 
            returned_var_one=var_one,
            returned_var_two=var_two,
            returned_var_three=var_three,
            returned_list=list_of_inputs))

    return(flask.render_template('input_values.html'))


@app.route('/images/')
def images():
    return flask.render_template('images.html')


@app.route('/bootstrap/')
def bootstrap():
    return flask.render_template('bootstrap.html')


@app.route('/classify_image/', methods=['GET', 'POST'])
def classify_image():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('classify_image.html'))

    if flask.request.method == 'POST':
        # Get file object from user input.
        file = flask.request.files['file']

        if file:
            # Read the image using skimage
            img = skimage.io.imread(file)

            # Resize the image to match the input the model will accept
            img = skimage.transform.resize(img, (28, 28))

            # Flatten the pixels from 28x28 to 784x0
            img = img.flatten()

            # Get prediction of image from classifier
            predictions = image_classifier.predict([img])

            # Get the value of the prediction
            prediction = predictions[0]

            return flask.render_template('classify_image.html', prediction=str(prediction))

    return(flask.render_template('classify_image.html'))




if __name__ == '__main__':
    app.run(debug=True)