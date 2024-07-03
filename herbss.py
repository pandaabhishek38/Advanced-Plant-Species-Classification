from flask import Flask, render_template, request

from numpy import loadtxt
from keras.models import load_model
import cv2
import matplotlib.pyplot as plt 
import numpy as np
from keras.preprocessing import image

leafmodel = load_model('model.h5')



app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():

    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images" + imagefile.filename
    imagefile.save(image_path)

    IMAGE_SIZE = (150, 150)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE) 
    #image.shape
    image = image / 255.0
    image = image.reshape(-1,150,150,3)
    predictions1 = leafmodel.predict(image)     # Vector of probabilities
    pred_labels1 = np.argmax(predictions1, axis = 1) # We take the highest probability
    result = pred_labels1[0]
    if result == 0:
        label="Arive-Dantu"
    elif result == 1:
        label="Basale"
    elif result == 2:
        label="Karanda"
    elif result == 3:
        label="Jasmine"
    elif result == 4:
        label="Mint"
    elif result == 5:
        label="Drumstick"
    else:
        label="Pomegranate"

    return render_template('index.html', prediction=label)



if __name__ == '__main__':
    app.run(port=3000, debug=True)    