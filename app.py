from flask import Flask, render_template, request
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from random import randint
import tensorflow_addons as tfa
from keras.utils import load_img, img_to_array

app = Flask(__name__)

#dic = {0 : 'NonDemented', 1 : 'VeryMildDemented' , 2 : 'MildDemented' , 3 : 'ModerateDemented' }
CLASSES = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
labels =dict(zip([0,1,2,3], CLASSES))


model=tf.keras.models.load_model('VGG16-MODEL')
#model = load_model('custom_cnn_model.h5')

#model.make_predict_function()

"""def predict_label(img_path):
    #work_dr = IDG(rescale = 1./255)
    #train_data_gen = work_dr.flow_from_directory(directory=img_path, target_size=(176,176), batch_size=1200, shuffle=False)
    #test_data , test_labels = train_data_gen.next()
    p = model.predict(test_data)
    return labels[np.argmax(p[0])]"""

#model.make_predict_function()

frst = 1

def predict_label(img_path):
	i = load_img(img_path, target_size=(176,176))
	i = img_to_array(i)/255.0
	i = i.reshape(1, 176,176,3)
	p = model.predict(i)
	return labels[np.argmax(p[0])]
    


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/test", methods=['GET', 'POST'])
def test():
	return render_template("predict1.html")

@app.route("/about")
def about_page():
	return "Analyse d'images IRM avec les techniques de l'apprentissage profond dans le cadre de la détection de la maladie d'Alzheimer."

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	global frst
	if request.method == 'POST':
		img = request.files['my_image']
        
		img_path = "static/" + img.filename	
		img.save(img_path)
        
		if frst == 1 :
			time.sleep(7)
			frst=2

		p = predict_label(img_path)
		

	return render_template("predict1.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(host='127.0.0.1', port=8000,debug = True)