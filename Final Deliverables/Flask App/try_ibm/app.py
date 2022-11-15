import joblib as joblib
from flask import Flask, request, render_template
import os
from tensorflow import python as tf_python
from tensorflow.python import keras
import numpy as np
from tensorflow.python.keras.models import load_model

app = Flask(__name__)

@app.route('/')
def hello_world(name="NO DATA"):
   return render_template('hello.html', name=name)

@app.route('/predict',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['PRICE']

      model = load_model('model1.h5')
      string = user
      string = string.split(',')
      x_input = [eval(i) for i in string]
      if (len(string) == 10):
          sc = joblib.load("scaler.save")
          x_input = sc.fit_transform(np.array(x_input).reshape(-1, 1))
          x_input = x_input.reshape((1, 10, 1))
          res = model.predict(x_input)
          val  = res[0][0]*100
          val = float("{:.2f}".format(val))
          return render_template('hello.html', name=val)
      else:
          val = 'Retry'
          return render_template('hello.html', name=val)

if __name__ == '__main__':
   app.run()