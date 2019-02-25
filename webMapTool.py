from pyproj import Proj, transform

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
    return img,bbox
