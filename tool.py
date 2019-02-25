import os 

import shutil
import cv2
from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D,MaxPooling2D, Dense, Dropout, BatchNormalization, Flatten
from keras.models import Model
from keras.preprocessing.image import img_to_array
import numpy as np



def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image


def create_batches(path,files):
    images = []
    for file in files:
        print("Pic:" + path +file)
        img = cv2.imread(path+file,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (75,75))
        img = img/255
        img = img_to_array(img)
        images.append(img)
    images = np.array(images).reshape(-1,75,75,3)
    return images

def saveImage(model,epochs,path,picType,label,fileName,pil_im,left,upper,right,lower):
    files = []
    for file in os.listdir(path + picType):
        if file.endswith('png') or file.endswith('jpg'):
            files.append(file)

    number_files = len(files)
    
    # if number_files >= 5:
    #     images =  create_batches(path+picType,files)
    #     retrainModel(model,epochs,images,number_files,label)
    #     for file in files:
    #         print("move file " + file)
    #         shutil.move(path+picType+file, path+"Trained/"+picType+file)

    print("Choose " + picType)

    pil_im.crop((left,upper,right,lower)).save(path +  picType + fileName)
 

    return True


def retrainModel(model,epochs,images,number_files,label):
    label = np.asarray(label*number_files)
    print("Retrain the model")
    model.fit(images,label,batch_size=number_files,epochs=epochs)
    print("Update model")
    return None

def vgg16_model(trainable=True):
    base_model = VGG16(False, "imagenet")
    train_from_layer = -2
    for layer in base_model.layers[:train_from_layer]:
        layer.trainable = False
        print("{} is not trainable".format(layer.name))
    for layer in base_model.layers[train_from_layer:]:
        #layer.trainable = True
        layer.trainable = False
        print("{} is trainable".format(layer.name))
    last_conv_layer = base_model.get_layer("block5_conv3")
    x = GlobalAveragePooling2D()(last_conv_layer.output)
    #x = Flatten()(last_conv_layer.output)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation="relu")(x)        
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(base_model.input, predictions)
    model.compile(optimizer="adadelta", loss='binary_crossentropy')
    return model

def classifyImage(model,tiles):
    satelliteIndex  = []
    count = 0
    for tile in tiles:
        try:
            prediction = model.predict(np.expand_dims(tile/255,axis=0))      
    #         predicted_class = np.argmax(prediction)
            predicted_class = np.round(prediction)
        
            if predicted_class ==0:
                count = count
    #             myimg = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
    #             cv2.imwrite( negFilePath + image_name+ "_img_"+str(count)+".png",myimg)
            if predicted_class ==1:
                print(count)
                satelliteIndex.append(count)
                
                #myimg = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
                #cv2.imwrite( "img_"+str(count)+".png",tile)      
            count+=1
        except: 
            #print("shape")
            traceback.print_exc()
            print(tile.shape)
    return satelliteIndex
