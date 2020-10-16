#ShengpingJiang- Face recognition model as a flask application

import pickle
import numpy as np
from flask import Flask, request

#model = None
app = Flask(__name__)


def load_model():
    global model
    # model variable refers to the global variable
    with open('face_model_file_frg', 'rb') as f:
        model = pickle.load(f)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['GET','POST'])
def get_prediction():
    dist_threshold = 0.4
    name=''
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        #data[0] means 1st {} in the JSON data [{..},{..}]. data[0]['encoding'] means 
        #the value of key 'encoding' in data[0]
        #print(type(data[0]['encoding']))
        #print(data[0]['encoding'])
        #The value of the key 'encoding' is a string '[-0.17077433  0.086519...]'
        str1 = data[0]['encoding']
        # str1[1:-1] from '[-0.17077433  0.086519...]' to '-0.17077433  0.086519...'. Remove brackets
        # np.fromstring changes a string '-0.17077433  0.086519...' to a numpy array 
        # [-0.17077433  0.086519...]
        encoding = np.fromstring(str1[1:-1], dtype=float, sep=' ')
        #print("ecoding type:", type(encoding))
        #print(encoding)
        
        # reshape(1,-1) change [-0.17077433  0.086519...] to [[-0.17077433 0.086519 0.04608656...]]
        xt = encoding.reshape(1,-1)
        #print('xt:', xt)
        closest_distance = model.kneighbors(xt, n_neighbors=1, return_distance=True)
        #print("closest_distance[0][0][0]:",closest_distance[0][0][0])
        if closest_distance[0][0][0] <= dist_threshold :
	# model.predict(xt) returns a string list ['name']
	# model.predict(xt)[0] returns 'name'
            name = model.predict(xt)[0]
            print('name:', name)
        else:
            name = "Unknown"
    elif request.method == 'GET':
        print("Shengping")
        
    return name


if __name__ == '__main__':
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=5000)

