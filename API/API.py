from __future__ import division, print_function
import os
import sqlite3
from datetime import datetime, date
from itertools import count
import getpass
from time import strftime

import numpy as np
# Keras
from sqlalchemy.sql.functions import current_timestamp
from werkzeug.exceptions import BadRequest
from keras.models import load_model
# Flask utils
from flask import Flask, request, render_template, url_for, redirect, session, send_from_directory
from werkzeug.utils import secure_filename
import cv2
from flask import Flask, redirect, render_template, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user

# Define a flask app
app = Flask(__name__)
# app.secret_key = "__privatekey__"

MODEL_PATH = 'my_h5_model2.h5'

# Load your trained model
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')


@app.errorhandler(404)
def page_not_found(error):
    return 'page not found', 404


@app.errorhandler(BadRequest)
def handle_bad_request(e):
    return 'bad request!', 400


def convertToBlobData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        blobData = file.read()
    return blobData


def model_predict(img_path, model):
    x = RGBAtoRGB(img_path)
    print(x.shape)
    x = np.asarray(x)
    print(x.shape)
    x = np.asarray([x])
    print(x.shape)
    preds = model.predict(x)
    return preds


def valid_register(name, emaill, password):
    print(name, emaill, password)
    con = sqlite3.connect("Leukmia.db")
    c = con.cursor()
    c.execute(f"SELECT * FROM Users WHERE Email = '{str(emaill)}'")
    rows = c.fetchall()
    print(rows)
    con.commit()
    if len(rows) != 0:
        return False
    else:
        # name = name.lower()
        # name = name.replace(" ", "")
        print(name)
        c.execute("INSERT INTO Users VALUES (?, ?, ?)", (emaill, password, name))
        con.commit()
        # email = emaill
        return True


def valid_login(emaill, password):
    print(emaill, password)
    con = sqlite3.connect("Leukmia.db")
    c = con.cursor()
    c.execute(f"SELECT * FROM Users WHERE Email = '{str(emaill)}' and Password = '{str(password)}'")
    rows = c.fetchall()
    print(rows)
    con.commit()
    if len(rows) < 1:
        return False
    else:
        return True


@app.route('/', methods=['POST', 'GET'])
def register():
    # userr = request.cookies.get('user')
    # print(userr)
    error = ""
    if request.method == 'POST':
        if valid_register(request.form['name'],
                          request.form['email'],
                          request.form['password']):
            email = request.form['email']
            response = redirect(url_for("index"))
            response.set_cookie('user', email)
            return response
        else:
            error = "Already Exist"
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return render_template('/register.html', error=error)


@app.route('/login', methods=['POST', 'GET'])
def login():
    error = ""
    if request.method == 'POST':
        if valid_login(request.form['email'],
                       request.form['password']):
            # global email
            email = request.form['email']
            response = redirect(url_for("index"))
            response.set_cookie('user', email)
            return response
        else:
            error = 'Invalid email/password'
    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return render_template('/login.html', error1=error)


@app.route('/home')  # , methods=['GET'])
def index():
    # Main page
    return render_template('/index.html')


@app.route('/allClass')  # , methods=['GET'])
def classAll():
    print("blablabalb")
    return render_template('/allClass.html')


@app.route('/result')  # , methods=['GET'])
def resFun(result=""):
    return render_template('/result.html', result=result)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    con = sqlite3.connect("Leukmia.db")
    c = con.cursor()
    # classes = {0: "ALL", 1: "AML", 2: "CLL", 3: "CML", 4: "Normal"}
    classes = ["ALL", "AML", "CLL", "CML", "Normal"]
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print(basepath)
        file_path = os.path.join(
            basepath, 'static/images', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)

        result = ""
        for i in range(5):
            if i != 4:
                result += classes[i] + ": " + str(round(preds[0][i] * 100, 2)) + "%, "
            else:
                result += classes[i] + ": " + str(round(preds[0][i] * 100, 2)) + "%"
        # path = send_from_directory("uploads", secure_filename(f.filename))
        # print(path)
        # user = getpass.getuser()
        # print(user)
        # c.execute(f"SELECT * FROM Users WHERE Username = '{user}'")
        # con.commit()
        # email = c.fetchall()
        email = request.cookies.get("user")
        print(email)
        print("something")
        print(email, result)
        print("after something")
        imageBlob = convertToBlobData(file_path)
        print(imageBlob)
        c.execute("INSERT INTO Images ('Email', 'image', 'Result')VALUES(?, ?, ?)", (email, imageBlob, result))
        con.commit()
        return render_template('/result.html', result=result, imageFile='/static/images/' + secure_filename(f.filename))
    return "Error Occurred"
    # return render_template("/error.html")


def RGBAtoRGB(img):
    imgg = cv2.imread(img)
    print(imgg.shape)
    imgg = cv2.resize(imgg, (124, 124))
    print(imgg.shape)
    imgg = cv2.cvtColor(imgg, cv2.COLOR_BGRA2RGB)
    print(imgg.shape)
    return imgg


if __name__ == '__main__':
    app.run(debug=True)

# Process your result for human
# pred_class = np.argmax(preds, axis=1)  # Simple argmax
# pred_class = decode_predictions(preds, top=1)  # ImageNet Decode
# print(pred_class)
# print(classes[pred_class[0]])
# result = str(classes[pred_class[0]])  # Convert to string
