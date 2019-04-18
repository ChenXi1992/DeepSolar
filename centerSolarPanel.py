import webMapTool
from tqdm import tqdm
from owslib.wms import WebMapService
from PIL import Image
import io
import os

imagePath = 'Test/tiff/'
layer =  'nw_dop_rgb'
img_format = "image/tiff"
style='default'
wms = WebMapService('https://www.wms.nrw.de/geobasis/wms_nw_dop', version='1.1.1')
import json
import re
from pyproj import Proj, transform

def downloadImg(imagePath,gps_x,gps_y,x_meters,y_meters,resolution):
    loc = (gps_x, gps_y) 
    print(loc)
    locs = webMapTool.slide_location(loc,xmeters=x_meters,ymeters=y_meters,xtimes=1,ytimes=1)
    for loc in tqdm(locs):
        #print("x_meters is {}, y_meters is {}, image format is {}, loc is {}".format(x_meters,y_meters,img_format,loc))
        img, bbox_m = webMapTool.img_selector(wms,layer,img_format,loc, styles=style , x_meters=x_meters,y_meters=y_meters, x_pixels=resolution,y_pixels =resolution)
        mybyteimg = img.read()
        image1 = Image.open(io.BytesIO(mybyteimg))

    imgName = "_x_"+str(gps_x) +"_y_"+str(gps_y)+"_range_"+str(x_meters)+"_resolution_"+ str(resolution)+ ".tiff"
    os.makedirs(os.path.dirname(imagePath), exist_ok=True)
    image1.save(imagePath + imgName)

outProj = Proj(init='epsg:3857') # https://epsg.io/3857, basically it allows me to specify things in meters..
inProj = Proj(init='epsg:4326') # https://epsg.io/4326

jsonFile = "annotations/clean/data_6_clean.json"
jsonFile = open(jsonFile)
data = json.load(jsonFile)

for i in range(len(data)):
    test = data[i]
    result = re.findall(r'\d+',test['url'])
    x = result[0] + "." + result[1]
    y = result[2] + "." + result[3]    
    poi =  test['POIs']
    points = json.loads(poi)
    # Shift x & y ∈  (-2/x , 2/x) 
    shift_x = 0   
    shift_y = 0 
    for point in points:
        dis_x = point['x'] * 0.1 
        dis_y = point['y'] * 0.1
        x1, y1 = transform(inProj,outProj,x,y)
        p = Proj("+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=6378137 +b=6378137 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs")
        new_x = x1 - 25 + dis_x + shift_x  
        new_y = y1 + 25 - dis_y + shift_y 
        lon, lat = p(new_x,new_y, inverse=True)
        downloadImg(imagePath,lon,lat,50,50,500)