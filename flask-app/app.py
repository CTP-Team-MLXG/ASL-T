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

import werkzeug.utils

app = flask.Flask(__name__, template_folder='templates')


# ASL File Paths
path_to_asl_categories = 'models/CATEGORIES.pickle'
path_to_asl_classifier = 'models/cnn1'

# Saving to variables for usage
try:
    asl_cnn_classifier = tensorflow.keras.models.load_model(path_to_asl_classifier)
except EOFError as e:
    print(e)

with open(path_to_asl_categories, 'rb') as f:
    ASL_CATEGORIES = pickle.load(f)


# For Uploading Images
UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key" 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# Create Upload directory
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':

        if 'file' not in flask.request.files:
            return flask.redirect(flask.request.url)

        # print(dir(flask.request))
        # Get file object from user input.
        file = flask.request.files['file']

        if file.filename == '':
            return flask.redirect(flask.request.url)

        if file and allowed_file(file.filename):
            # Save the image to the backend static folder
            filename = werkzeug.utils.secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            
            # Read image file
            img = cv2.imread(path)

            cv2.imwrite(path, cv2.resize(img, (300, 300)))
            # Read image file string data
            # filestr = file.read()
            
            # Convert string data to np arr
            # npimg = np.frombuffer(filestr, np.uint8)
            # Convert np arr to image
            # img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

            # Resize the image to match the input the model will accept
            img = cv2.resize(img, (64, 64))
            # Reshape the image into shape (1, 64, 64, 3)
            img = np.asarray([img])

            # Get prediction of image from classifier
            prediction = np.argmax(asl_cnn_classifier.predict(img), axis=-1)

            # Get the value at index of CATEGORIES
            prediction = ASL_CATEGORIES[prediction[0]]
            
            return flask.render_template('main.html', 
                prediction=prediction,
                filename=filename)
        else:
            return flask.redirect(flask.request.url)

        # if file:
        #     # Read image file string data
        #     filestr = file.read()
            
        #     # Convert string data to np arr
        #     npimg = np.frombuffer(filestr, np.uint8)
        #     # Convert np arr to image
        #     img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        #     # Resize the image to match the input the model will accept
        #     img = cv2.resize(img, (64, 64))
        #     # Reshape the image into shape (1, 64, 64, 3)
        #     img = np.asarray([img])

        #     # Get prediction of image from classifier
        #     prediction = np.argmax(asl_cnn_classifier.predict(img), axis=-1)

        #     # Get the value at index of CATEGORIES
        #     prediction = ASL_CATEGORIES[prediction[0]]

        #     return flask.render_template('main.html', 
        #         prediction=prediction,
        #         image=file)

    return(flask.render_template('main.html'))


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


@app.route('/about/')
def about():
    return flask.render_template('about.html')


@app.route('/contributors/')
def contributors():
    return flask.render_template('contributors.html')

    

if __name__ == '__main__':
    app.run(debug=True)