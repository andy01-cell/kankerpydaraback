import json
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from flask import  Flask, request, json,jsonify
import pydicom
from PIL import Image
import urllib.request
from werkzeug.utils import secure_filename
from flask_cors import CORS
import base64
from requests_toolbelt import MultipartEncoder
import pathlib

categories = ["stadium 1", "stadium 2", "stadium 3", "stadium 4", "normal"]

test = []

# dirimg = final_image.save(os.path.join(app.config['UPLOAD_CLASIFER'], "clasifier.jpg"))
dirimg = "E:\\dataset deep learning\\Stadium kanker payudara\\Dataset\\stadium 4\\ancfyflrmjkqfiqwgflj.jpg"
print("dir = ", dirimg)
kanker_img = cv2.imread(dirimg, 0)
kanker_img = cv2.resize(kanker_img, (50, 50))
image = np.array(kanker_img).flatten()

test.append(image)
print("image = ", image)
print("test= ", test)

#
# data = []
# dir= "E:\\dataset deep learning\\Stadium kanker payudara\\Dataset"
#
# for category in categories:
#   path = os.path.join(dir,category)
#   label = categories.index(category)
#
#   for img in os.listdir(path):
#     imgpath = os.path.join(path,img)
#     kanker_img=cv2.imread(imgpath,0)
#     try:
#       kanker_img=cv2.resize(kanker_img,(50,50))
#       image = np.array(kanker_img).flatten()
#       data.append([image,label])
#
#     except Exception as e:
#       pass
#
# pick_in = open('data.pickle','wb')
# pickle.dump(data,pick_in)
# pick_in.close ()
#
pick_in = open('data.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

# random.shuffle(data)
features = []
labels = []

for feature, label in data:
  features.append(feature)
  labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.20, stratify=labels)
model = SVC(C=50, kernel='rbf', gamma='scale')
model.fit(xtrain, ytrain)


# print("data = ", len(data))
# print("training = ", len(xtrain))
# print("testing = ", len(xtest))



print("p = ", features[0])

prediction = model.predict(test)
accuracy = model.score(features, labels)

print('Accuracy : ', accuracy)
print('Prediksi : ', categories[prediction[0]])

mypydr = xtest[0].reshape(50, 50)

plt.imshow(mypydr)
plt.show()
