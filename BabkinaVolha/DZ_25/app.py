import datetime
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def read_home():
    return render_template('home.html')

@app.route('/about/')
def about():
    return render_template('about.html')

app.run(host='0.0.0.0', port=5000,debug=True)