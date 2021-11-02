import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #Render Output In Web Page
    in_features = [float(x) for x in request.form.values()]
    final_features = [np.array(in_features)]
    s = ['_','XXS','XS','S','M','L','XL','XXL','XXXL']
    in_size = model.predict(final_features)[0]
    size = s[round(in_size)]

    return render_template('result.html',prediction='Cloth Size is: {}'.format(size))

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000,debug=True)

