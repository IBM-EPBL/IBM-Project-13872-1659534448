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
      user = str(user)
      temp = user.split('-')
      temp.append(temp[0])
      temp.pop(0)
      temp1 = ""
      user = ""
      for x in temp:
         temp1+=x

      for x in temp1:
         user+=x+","

      user = user[:-1]
      model = load_model('model1.h5')
      string = user
      string = string.split(',')
      for x in range(0, 82):
         string.append('1')
      x_input = [eval(i) for i in string]
      sc = joblib.load("scaler.save")
      x_input = sc.fit_transform(np.array(x_input).reshape(-1, 1))
      x_input = x_input.reshape((1, 90, 1))
      res = model.predict(x_input)
      val  = res[0][0]*1000
      val = float("{:.2f}".format(val))
      return render_template('hello.html', name=val)

if __name__ == '__main__':
   app.run()