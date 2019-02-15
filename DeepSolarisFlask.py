# USAGE
# Start the server:
#   python run_keras_server.py
# Submit a request via cURL:
#   curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#   python simple_request.py

# import the necessary packages
from keras.applications import ResNet50

from keras.applications import imagenet_utils

import flask
import io

from keras.applications import VGG16
from keras.layers import GlobalAveragePooling2D,MaxPooling2D, Dense, Dropout, BatchNormalization, Flatten
from keras.models import Model
import tensorflow as tf

from flask import render_template, jsonify,url_for, request


# For download the pic  & Cutting the image
from owslib.wms import WebMapService
import io
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import traceback
from tqdm import tqdm
from pyproj import Proj, transform
import cv2
from math import ceil
import os


import tool 
import webMapTool

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = tool.vgg16_model(False)
model.load_weights('static/vgg16_3t_wmp_wr_aachen__06_0.89.hdf5')
graph = tf.get_default_graph()
country = "Germany"
wms = WebMapService('https://www.wms.nrw.de/geobasis/wms_nw_dop', version='1.1.1') # Germany


 # Netherlands, WebMapService('https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms?&request=GetCapabilities', version='1.1.1')

layer =  'nw_dop_rgb'
img_format = "image/tiff"
style='default'
x_meters = 500
y_meters = 500
x_pixels = 1000
y_pixels = 1000
imgPath = "static/img/"
imgName = "download.tiff"
cutSize = 75
epochs = 10 
    

@app.route('/')
def display_web():
    return render_template('template.html')

@app.route("/downloadPic", methods = ["POST","GET"])

def downloadImage():
    gps_x = float(request.args.get('gps_x') )
    gps_y = float(request.args.get('gps_y'))
    country = request.args.get('country')
    x_meters = float(request.args.get('x_range'))
    y_meters = float(request.args.get('y_range'))
    resolution = float(request.args.get('resolution'))

    print("country = {}, x = {}, y = {}, x_range = {}, y_range = {}".format(country,gps_x,gps_y,x_meters,y_meters))

    if country == 'Netherlands':
        wms = WebMapService('https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms?&request=GetCapabilities', version='1.1.1')
        layer = 'Actueel_ortho25'
    if country == 'Germany':
        wms = WebMapService('https://www.wms.nrw.de/geobasis/wms_nw_dop', version='1.1.1')
        layer = 'nw_dop_rgb'

    loc = (gps_x, gps_y) 
    locs = webMapTool.slide_location(loc,xmeters=x_meters,ymeters=y_meters,xtimes=1,ytimes=1)
    images = []

    for loc in tqdm(locs):

        print("x_meters is {}, y_meters is {}, image format is {}, loc is {}".format(x_meters,y_meters,img_format,loc))
        img = webMapTool.img_selector(wms,layer,img_format,loc, styles=style , x_meters=x_meters,y_meters=y_meters, x_pixels=resolution,y_pixels =resolution)
        print("Start download pics")
        mybyteimg = img.read()
        image = Image.open(io.BytesIO(mybyteimg))
        images.append(image)
        
    image1 = images[0]

    imgName = country+ "_x_"+str(gps_x) +"_y_"+str(gps_y)+"_range_"+str(x_meters)+"_resolution_"+ str(resolution )+ ".tiff"

    image1.save(imgPath+imgName)

    pngPic =  cv2.imread(imgPath + imgName)

    pngName = imgName[:-5]+".png"

    cv2.imwrite(imgPath + pngName, pngPic)

    return jsonify({'url':imgPath+pngName})

@app.route("/detectSolarPanel",methods = ["POST","GET"])
def detectSolarPanel():

    url = request.args.get('url') 

    # image1=mpimg.imread(url)# for the moment I select manually   
    image1 = cv2.imread(url)
    print("Start cutting the pic to tiles")
    M = 75
    N = 75
    tiles = [image1[x:x+M,y:y+N] for x in range(0,image1.shape[0],M) for y in range(0,image1.shape[1],N)]
    # for i in range(0,len(tiles)):
    #     tiles[i]=cv2.cvtColor(tiles[i], cv2.COLOR_RGBA2RGB)
    # Do the classification 
    print("Start classification")
    satelliteIndex = tool.classifyImage(tiles)

    # Remark the pic and save it locally 
    for count in satelliteIndex:
        col = count % ceil(image1.shape[0] / M)
        row =    int(count / ceil(image1.shape[0] / M))
        cv2.circle(image1, (col*M+25,row*M+25), int(M/2), (0,0,255), thickness=10, lineType=8, shift=0) 
    # markedImg = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    markedUrl = url[:-4]+"_marked.png"
    cv2.imwrite(markedUrl, image1)

    return jsonify({'url':markedUrl})



@app.route("/labelData", methods=["POST","GET"])
def labelData():
    optionType = request.args.get('type') 
    x_val = float(request.args.get('click_X'))
    y_val = float(request.args.get('click_Y'))
    imgPath = request.args.get('img')
    print("img path is " + imgPath)
    fileName =  os.path.basename(imgPath)
    print("X is {}, Y is {}".format(x_val,y_val))

    fileName = fileName[:-4] + "_x_"+str(x_val)[:4] + "_y_" + str(y_val)[:4]+".png"

    pil_im = Image.open(imgPath)
    width,height = pil_im.size
    print("Image width is {}, height is {}".format(width,height))
    x_val = width * x_val
    y_val = height * y_val

    left = x_val - cutSize/2
    upper = y_val - cutSize/2
    right = x_val + cutSize/2
    lower = y_val + cutSize/2


    path = "static/label/"
    if optionType == "one":
        picType = "True_Positive/"
        label = [1]
        tool.saveImage(model,epochs,path,picType,label,fileName,pil_im,left,upper,right,lower)
    elif optionType == "two":
        picType = "False_Positive/"
        label = [0]
        tool.saveImage(model,epochs,path,picType,label,fileName,pil_im,left,upper,right,lower)

    elif optionType == "three":
        picType = "True_Negative/"
        label = [0] 
        tool.saveImage(model,epochs,path,picType,label,fileName,pil_im,left,upper,right,lower)

    elif optionType == "four":
        picType = "False_Negative/"
        label = [1]
        tool.saveImage(model,epochs,path,picType,label,fileName,pil_im,left,upper,right,lower)
        

    return jsonify({'results':"success"})
        


@app.route("/predict", methods=["POST","GET"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and pmsWMSLoadGetMapParamsepare it for classification
            image = tool.prepare_image(image, target=(75, 75))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            with graph.as_default():
                preds = model.predict(np.array(image))
                #results = imagenet_utils.decode_predictions(preds)
                data["predictions"] = []

                # loop over the results and add them to the list of
                # returned predictions
                #for (imagenetID, label, prob) in results[0]:
                #r = {"label": label, "probability": float(prob)}
                data["predictions"].append(preds[0].tolist())

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(host = 'localhost')
app.run()
