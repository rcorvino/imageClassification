import numpy as np
from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras import regularizers
import keras.backend as K
K.set_image_data_format('channels_last')
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.models import model_from_json
from keras.utils import to_categorical


import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import matplotlib.image as mpimg

from sklearn.metrics import accuracy_score
import os
import glob
from pathlib import Path

def getLabel(predictions):
    m = np.argmax(predictions)
    if m == 0:
        return "shaver"
    elif m == 1:
        return "smart-baby-bottle"
    elif m == 2:
        return "toothbrush"
    else:
        return "wake-up-light"

def read_model(model_name_json):
    """loads a json file and creates a model from it"""
    file_exists = False
    model = Model()
    if os.path.exists(model_name_json):
        json_file = open(model_name_json, 'r')
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        json_file.close()
        print(model_name_json+" loaded model from disk")
        file_exists = True
    return file_exists, model

def read_parameters(model, model_name_h5):
    """load weights into new model"""
    #print(model_name_h5)
    file_exists = False
    if os.path.exists(model_name_h5):
        model.load_weights(model_name_h5)
        print("weights of "+model_name_h5+" loaded model from disk")
        file_exists = True
    return file_exists


class ensemble:
    """ensemble of multiple models"""

    def __init__(self):
        self.input_models_list = []
        self.models_list = []

    def readall(self):
        model_file_names = glob.glob("models/*.json")
        print(model_file_names)
        for model_name in model_file_names:
            params_name = model_name.replace(".json", ".h5")
            model_exist, model = read_model(model_name)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            read_parameters(model, params_name)
            self.input_models_list.append((model, model_name))

    def predict(self, XY):
        """
        returns the addition of models predictions
        for XY tuple, where Y is the expected predictions for input X
        ------------------
        Parameters
        ------------------
        models_list list of models
        """
        a=0
        for model in self.models_list:
            a+=model.predict(XY.X)
        a= np.argmax(a, axis=1)
        b = to_categorical(a)
        return b

    def pPrint(self, XY, predictions):
        for idx in range(len(predictions)):
            print(XY.paths[idx]+"   -    "+getLabel(predictions[idx]))

    def evaluate(self, XY224, XY299):
        sel=""
        for m in self.input_models_list:
            model = m[0]
            X= XY224.X
            Y= XY224.Y
            if model.name.find("inception_v3")>0:
                X= XY299.X
                Y= XY299.Y
            print(model.name)
            scores = model.evaluate(X, Y, verbose=0)
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            if scores[1]*100>70 :
                print(m[1])
                sel +=m[1]+"\n"
                self.models_list.append(model)
        with open("selected", "w") as selected:
                selected.write(sel)

    def from_file(self):
        with open("selected", "r") as selected:
            files = selected.readlines()

        for f in files:
            f=f.replace("\n","")
            params_name = f.replace(".json", ".h5")
            model_exist, model = read_model(f)
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            read_parameters(model, params_name)
            self.models_list.append(model)



class XY:
    """call of input output set of images and their location on disk"""
    def __init__(self, it):
        self.X,self.Y = it.next()
        index_array = it.index_array.tolist()
        self.paths = [it.filepaths[idx] for idx in index_array]
        print(self.paths)


class network:
    """
    class containing a network which has a
    model: chosen among vgg16, resnet50, inceptionv3
    data flows: train_it, val_it, test_it
    """

    def __init__(self, name, W=224, H=224, batch_size=16, regularizer=0.001):
        self.name = name
        self.regularizer = regularizer
        self.batch_size = batch_size
        self.train_it = self.dataGenerator("data/train/", True, W, H, batch_size)
        self.val_it = self.dataGenerator("data/validation/", False, W, H, batch_size)
        self.test_it = self.dataGenerator("data/test/", False, W, H, batch_size)
        self.model = self.config_model()

    def dataGenerator(self, directory_path, plus, W, H, batch_size):
        """
        provides data generator with data aigmentation
        """
        if(plus):
            datagen = ImageDataGenerator(
            horizontal_flip=True, vertical_flip=True,
            rotation_range=20,
            brightness_range=[0.2,1.0],
            zoom_range=[0.5,1.0])
        else:
            datagen = ImageDataGenerator(
            horizontal_flip=True, vertical_flip=True,
            brightness_range=[0.2,1.0])
        return datagen.flow_from_directory(
        directory_path, target_size=(W,H),
        batch_size=batch_size, class_mode='categorical',
        follow_links=True)

    def read_model_from_disk(self):
        """construct the path of the model json file
        calls the loader"""
        model_name_json = "models/modified_"+self.name+str(self.regularizer)+".json"
        return read_model(model_name_json)

    def config_model(self):
        """ Load the base model,
        pops the last layers and
        inserts a 4-classes classification layer.
        Freezes weights of all the pre-existing layers """
        file_exists, model = self.read_model_from_disk()
        if not(file_exists):
            if self.name == 'vgg16':
                model = VGG16()
            elif self.name == 'resnet50':
                model = ResNet50()
            elif self.name == 'inceptionv3':
                model = InceptionV3()
            model.layers.pop()
            print("model: "+model.name)
            print("  #layers: "+ str(len(model.layers)))
            for layer in model.layers:
                layer.trainable = False
            output = Dense(4, activation='softmax',
                       kernel_regularizer=regularizers.l1(self.regularizer))(model.layers[-1].output)
            model = Model(inputs=model.inputs, outputs=output, name="modified_"+model.name)
        return model

    def read_parameters_from_disk(self):
        """construct the path of the model h5 file
        calls the loader"""
        model_name_h5 = "models/"+self.model.name+str(self.regularizer)+".h5"
        return read_parameters(self.model, model_name_h5)

    def compile_and_fit(self):
        """Compile the model and fit it or load the
        h5 file if it exists"""
        print("model: "+self.model.name)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        if not(self.read_parameters_from_disk()):
            self.model.fit(
                self.train_it, epochs=5, steps_per_epoch=2048//self.batch_size,
                validation_data=self.val_it, validation_steps=1024//self.batch_size)
    def serialize(self):
        file_exists, _model = self.read_model_from_disk()
        if not(file_exists):
            # serialize model to JSON
            model_json = self.model.to_json()
            model_name_jason = "models/"+self.model.name+str(self.regularizer)+".json"
            model_name_h5 = "models/"+self.model.name+str(self.regularizer)+".h5"
            with open(model_name_jason, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            self.model.save_weights(model_name_h5)
            print("Saved "+ self.model.name+" to disk")

    def evaluate(self):
        print(self.model.name+" accuracy")
        test_loss = self.model.evaluate_generator(self.test_it)
        val_loss = self.model.evaluate_generator(self.val_it)
        train_loss = self.model.evaluate_generator(self.train_it)
        print("Test = {0:.2f} ".format(round(test_loss[1],2))+
              "Validation = {0:.2f} ".format(round(val_loss[1],2))+
              "Train = {0:.2f} ".format(round(train_loss[1],2)))
        return round(test_loss[1],2), round(val_loss[1],2), round(train_loss[1],2)

    def evaluate_on(self, XY):
        """
        returns accuracy of model on XY.X, XY.Y tuple
        ------------------
        Parameters
        ------------------
        XY.X set on which evaluate the ensamble
        XY.Y set of expected predictions for XY.X
        """
        scores = self.model.evaluate(XY.X, XY.Y, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
        return scores[1]

    def predict_on(self, XY):
        """
        returns model's predictions
        for XY
        """
        a = self.model.predict(XY.X)
        #print(a)
        a= np.argmax(a, axis=1)
        b = to_categorical(a)
        return b

    def plot_mispredictions(self, XY, predictions):
        """
        plots pictures of mis-predicted cases
        ------------------
        Parameters
        ------------------
        XY.X input
        XY.Y expected predictions
        predictions predictions (Yhat)
        """
        c=XY.Y-predictions
        max=np.argmax(c,axis=0)
        fig=plt.figure(figsize=(24, 24))
        i=1
        for elem in max:
            if elem!=0:
                fig.add_subplot(2, 2, i)
                plt.imshow(XY.X[elem].astype('uint8'))
                i+=1
        plt.show()

    def labeled_predictions(self, XY, predictions):
        """
        prints labels corresponding to the predictions
        """
        labels = {0:"shaver", 1:"smart-baby-bottle", 2:"toothbrush", 3:"wake-up-light"}
        max_predictions = np.argmax(predictions, axis=1)
        max_Y = np.argmax(XY.Y, axis=1)
        pred_for_expected =[]
        for i in range(len(max_predictions)):
            pred_for_expected.append((labels[max_predictions[i]], labels[max_Y[i]]))
        print(pred_for_expected)
        #print(Y)


    def pprint(self, XY, predictions):
        for idx in range(len(predictions)):
            print(XY.paths[idx]+": "+getLabel(predictions[idx]))


def explore():
    #eplore possible models and regularizers
    possible_models=["vgg16", "resnet50","inceptionv3"]
    regularizers = [0.001, 0.002, 0.01, 0.05, 0.1]

    for mod in possible_models:
        for reg in regularizers:
            if mod == "inceptionv3":
                W=H=299
            else:
                W=H=224
            base_model = network(mod, W, H, regularizer=reg)
            base_model.compile_and_fit()

def main():
    if 0:
        datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True,brightness_range=[0.2,1.0])
        en = ensemble()
        en.readall()
        test_it_224 = datagen.flow_from_directory(
        "data/test", target_size=(224,224),
        batch_size=16, class_mode='categorical',
        follow_links=True)
        test_it_299 = datagen.flow_from_directory(
        "data/test", target_size=(299,299),
        batch_size=16, class_mode='categorical',
        follow_links=True)
        XY0_224 = XY(test_it_224)
        XY0_299 = XY(test_it_299)

        en.evaluate(XY0_224, XY0_299)
        test_it_224 = datagen.flow_from_directory(
        "data/test", target_size=(224,224),
        batch_size=16, class_mode='categorical',
        follow_links=True)
        XY0_224 = XY(test_it_224)
        predictions = en.predict(XY0_224)
        en.pPrint(XY0_224, predictions)



    datagen = ImageDataGenerator()
    imgs = datagen.flow_from_directory('.', classes=['validation'], target_size=(224,224))
    XY0 = XY(imgs)
    if len(XY0.paths)>0:
        en0 = ensemble()
        en0.from_file()
        predictions = en0.predict(XY0)
        en0.pPrint(XY0, predictions)
    else:
        print("directory *validation* is empty")

if __name__ == "__main__":
    main()
