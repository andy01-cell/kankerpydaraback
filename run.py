import base64
from flask import Flask, request, json,jsonify
import os
import numpy as np
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pydicom
from PIL import Image
from werkzeug.utils import secure_filename
from flask_cors import CORS
import cv2

my_awesome_app = Flask(__name__)
CORS(my_awesome_app)

my_awesome_app.secret_key = "caircocoders-ednalan"

UPLOAD_FOLDER = 'static/uploads'
my_awesome_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# my_awesome_app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

UPLOAD_CLASIFER = 'static/clasifer'
my_awesome_app.config['UPLOAD_CLASIFER'] = UPLOAD_CLASIFER

ALLOWED_EXTENSIONS = set(['dcm'])


@my_awesome_app.route('/')
def hello_world():
    resp = jsonify({'message': 'Ping'})
    resp.status_code = 201
    return resp

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@my_awesome_app.route('/upload', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'files' not in request.files:
        resp = jsonify({'message': 'No file part in the request'})
        resp.status_code = 400
        return resp

    files = request.files.getlist('files')

    errors = {}
    success = False

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(my_awesome_app.config['UPLOAD_FOLDER'], filename))
            success = True

            # ----convert image----
            im = pydicom.dcmread(os.path.join(my_awesome_app.config['UPLOAD_FOLDER'], filename))
            im = im.pixel_array.astype(float)

            rescaled_image = (np.maximum(im, 0) / im.max()) * 255  # float pixel
            final_image = np.uint8(rescaled_image)  # integer pixel

            final_image = Image.fromarray(final_image)
            print("jpg = ", final_image)
            # final_image.show()
            # final_image.save("1_2.jpg")


            # ----klasifikasi----
            # dir= "E:\\dataset deep learning\\Stadium kanker payudara\\Dataset"
            #

            categories = ["stadium 1", "stadium 2", "stadium 3", "stadium 4", "normal"]

            test = []

            dirimg = final_image.save(os.path.join(my_awesome_app.config['UPLOAD_CLASIFER'], "clasifier.jpg"))
            dirimg = os.path.join("static/clasifer/clasifier.jpg")
            print("dir = ", dirimg)
            kanker_img = cv2.imread(dirimg, 0)
            kanker_img = cv2.resize(kanker_img, (50, 50))
            image = np.array(kanker_img).flatten()

            test.append(image)
            print("image = ", image)
            print("test= ", test)

            #
            data = []
            dir= "E:\\dataset deep learning\\Stadium kanker payudara\\Dataset"
            categories = ["stadium 1", "stadium 2", "stadium 3", "stadium 4"]

            for category in categories:
              path = os.path.join(dir,category)
              label = categories.index(category)

              for img in os.listdir(path):
                imgpath = os.path.join(path,img)
                kanker_img=cv2.imread(imgpath,0)
                try:
                  kanker_img=cv2.resize(kanker_img,(50,50))
                  image = np.array(kanker_img).flatten()
                  data.append([image,label])

                except Exception as e:
                  pass

            pick_in = open('data.pickle','wb')
            pickle.dump(data,pick_in)
            pick_in.close ()

            pick_in = open('data.pickle', 'rb')
            data = pickle.load(pick_in)
            pick_in.close()

            random.shuffle(data)
            features = []
            labels = []

            for feature, label in data:
                features.append(feature)
                labels.append(label)

            xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.20, stratify=labels)
            model = SVC(C=50, kernel='rbf', gamma='scale')
            model.fit(xtrain, ytrain)

            #-------------------------------------

            print("data = ", len(data))
            print("training = ", len(xtrain))
            print("testing = ", len(xtest))
            print("p = ", features[0])

            prediction = model.predict(test)
            accuracy = model.score(features, labels)

            print('Accuracy : ', accuracy)
            print('Prediksi : ', categories[prediction[0]])

            mypydr = xtest[0].reshape(50, 50)


            # plt.imshow(mypydr)
            # plt.show()

            # filekan = request.files[]
#             with open("kanker.jpg", "rb") as image_file:
#                 encoded_string = base64.b64encode(image_file.read())
#
            return {"akurasi": accuracy, "prediksi":categories[prediction[0]]}


        else:
            errors['message'] = 'File type is not allowed'

    if success and errors:
        errors['message'] = 'File(s) successfully uploaded'
        resp = jsonify(errors)
        resp.status_code = 500
        return resp
    if success:
        resp = jsonify({'message': 'Files successfully uploaded'})
        resp.status_code = 201
        return resp
    else:
        resp = jsonify(errors)
        resp.status_code = 500
        return resp

@my_awesome_app.route('/getfile')
def getfile():
    with open("kanker.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    print("imags", image_file)
    return image_file


if __name__ == '__main__':
    my_awesome_app.run()
