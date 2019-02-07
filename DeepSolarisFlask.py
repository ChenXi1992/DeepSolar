# USAGE
# Start the server:
#   python run_keras_server.py
# Submit a request via cURL:
#   curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#   python simple_request.py

# import the necessary packages
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
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







# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

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
    return Model(base_model.input, predictions)


model = vgg16_model(False)
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

def slide_location(loc,xmeters,ymeters,xtimes,ytimes):
    outProj = Proj(init='epsg:3857') # https://epsg.io/3857, basically it allows me to specify things in meters.
    inProj = Proj(init='epsg:4326') # https://epsg.io/4326
    x, y = transform(inProj,outProj,loc[0],loc[1])
    locs = []
    #in GPS
    #locs.append(loc)
    for i in range(xtimes):
        for j in range(ytimes):
            x = x + xmeters*i
            y = y + ymeters*j
            x_gps,y_gps=transform(outProj,inProj,x,y)
            loc = (x_gps,y_gps)
            locs.append(loc)
    return locs


def img_selector(wms,layer,img_format,loc, styles=None , x_meters=1000,y_meters=1000, x_pixels=5000,y_pixels =5000):
    outProj = Proj(init='epsg:3857') # https://epsg.io/3857, basically it allows me to specify things in meters..
    inProj = Proj(init='epsg:4326') # https://epsg.io/4326
    x, y = transform(inProj,outProj,loc[0],loc[1])
    region_size = (x_meters, y_meters)
    print("x:{},y:{},xmeter:{},ymeters:{}".format(x,y,x_meters,y_meters))
    xupper = int(round(x - region_size[0] / 2))
    xlower = int(round(x + region_size[0] / 2))
    yupper = int(round(y - region_size[1] / 2))
    ylower = int(round(y + region_size[1] / 2))
    bbox = (xupper, yupper, xlower, ylower)
    if not styles==None:
        print("*****************")
        print("wms:{}, layers:{}, bbox:{}, img_format:{}".format(wms,layer,bbox,img_format))
        print("*****************")
        img = wms.getmap(layers=[layer], styles=['default'], srs='EPSG:3857',
                     bbox=bbox, 
                     size=(x_pixels, y_pixels), format=img_format, transparent=True)
        print("load img info")
    else:

        img = wms.getmap(layers=[layer], srs='EPSG:3857',
                     bbox=bbox, 
                     size=(x_pixels, y_pixels), format=img_format, transparent=True)
    return img

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    print("load_model")
    
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

def classifyImage(tiles):
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
                #cv2.imwrite( posFilePath + image_name+ "_img_"+str(count)+".png",myimg)      
            count+=1
        except: 
            #print("shape")
            traceback.print_exc()
            print(tile.shape)
    return satelliteIndex


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

    print("country = {}, x = {}, y = {}, x_range = {}, y_range = {}".format(country,gps_x,gps_y,x_meters,y_meters))

    if country == 'Netherlands':
        wms = WebMapService('https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms?&request=GetCapabilities', version='1.1.1')
        layer = 'Actueel_ortho25'
    if country == 'Germany':
        wms = WebMapService('https://www.wms.nrw.de/geobasis/wms_nw_dop', version='1.1.1')
        layer = 'nw_dop_rgb'

    loc = (gps_x, gps_y) 
    locs = slide_location(loc,xmeters=x_meters,ymeters=y_meters,xtimes=1,ytimes=1)
    images = []

    for loc in tqdm(locs):

        print("x_meters is {}, y_meters is {}, image format is {}, loc is {}".format(x_meters,y_meters,img_format,loc))
        img = img_selector(wms,layer,img_format,loc, styles=style , x_meters=x_meters,y_meters=y_meters, x_pixels=x_pixels,y_pixels =y_pixels)
        print("Start download pics")
        mybyteimg = img.read()
        image = Image.open(io.BytesIO(mybyteimg))
        images.append(image)
        
    image1 = images[0]

    imgName = country+ "_x_"+str(gps_x) +"_y_"+str(gps_x)+"_range_"+str(x_meters)+".tiff"

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
    satelliteIndex = classifyImage(tiles)


    # Remark the pic and save it locally 
    for count in satelliteIndex:
        col = count % ceil(image1.shape[0] / M)
        row =    int(count / ceil(image1.shape[0] / M))
        cv2.circle(image1, (col*M+25,row*M+25), int(M/2), (0,0,255), thickness=10, lineType=8, shift=0) 
    # markedImg = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    markedUrl = url[:-4]+"_marked.png"
    cv2.imwrite(markedUrl, image1)



    return jsonify({'url':markedUrl})

@app.route("/predict", methods=["POST"])
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
            image = prepare_image(image, target=(75, 75))

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
