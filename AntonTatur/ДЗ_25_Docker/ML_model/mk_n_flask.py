import pickle
from flask import Flask, request
import os

model = pickle.load(open("Model.pkl", 'rb'))

# root = str(os.path.dirname(os.path.realpath(__file__)))
# model_root = root + "/Model.pkl"

# model = pickle.load(open(model_root, 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    sepal_length = request.args.get('sepal_length')
    sepal_width=request.args.get('sepal_width')
    a=[[sepal_length, sepal_width]]
    b=model.predict(a)
    if b==[0]:
        return '''<h1>Тип цветка - Setosa </h1>'''
    
    if b==[1]:
        return '''<h1>Тип цветка - Versicolour </h1>'''

    if b==[2]:
        return '''<h1>Тип цветка - Virginica </h1>'''

app.run(host='0.0.0.0', port=5000,debug=True)