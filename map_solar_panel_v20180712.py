#! /usr/bin/env python
 
import datetime 
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol

import pandas as pd


Fread= True #False
Mapprov=True
Mapall= False
Mapcluster = False


if Fread:

    opath='/home/bigdata07/DeepSol/fig/'
    ipath= '/home/bigdata07/DeepSol/data/'
    fadd= 'bag-adressen-v20180610.csv' #'inspireadressen.csv'
    df_add= pd.read_csv(os.path.join(ipath,fadd),sep=';')

    fpir='solarpanels.csv'
    df_pir= pd.read_csv(os.path.join(ipath,fpir),sep=',', header=None, names=['postcode','huisnummer','Idate'])
    df_pir['huisletter'] = df_pir.huisnummer.str.split('-').str.get(1)
    df_pir['huisnummer'] = df_pir.huisnummer.str.split('-').str.get(0)
    df_pir['huisnummer'] = df_pir.huisnummer.str.split(' ').str.get(0)
    df_pir['huisnummer'] =  df_pir['huisnummer'].astype(np.int64)
    df_pir['Idate']= pd.to_datetime(df_pir['Idate'],format='%Y%m%d', errors='coerce')
    
#endif

df_PV = df_pir.merge(df_add, how='inner', on=['postcode', 'huisnummer','huisletter'] ) 

if Mapcluster:

   import folium
   from folium.plugins import MarkerCluster
   from folium.plugins import FeatureGroupSubGroup
   from folium.features import CustomIcon
   from folium.plugins import BoatMarker
   print(folium.__version__)
   
   def coldef(iyear):
       coldict = {2010:'red', 2011:'blue',
	            2012:'cadet blue',2013:'darkred',
		    2014:'lightred',2015:'orange',
		    2016:'lightgreen',2017:'darkblue',
		    2018:'lightblue',2019:'purple',
	            2020:'darkpurple'}
       try:
          colval = coldict[iyear]
       except:
          colval = 'lightgray'
       return colval
   colormap = bcm.linear.Set3.scale(2010,2020)
   macarte = folium.Map(location=[51.437,5.478], zoom_start=8)
   mc = MarkerCluster()
   #creating sub-group for each year
   for iyear in np.unique(df_PV['Idate'].dt.year):
      eval('g'+iyear+' = FeatureGroupSubGroup(mcg, '+str(iyear)+')')
      
      #creating a Marker for each point in df_sample. Each point will get a popup with their zip
      for index,row in df_PV[df_PV['Idate'].dt.year==iyear].iterrows():
           mc.add_child(folium.CircleMarker(
	             location =[row['lat'], row['lon']],
                     radius=5,
                     color= colormap(int(row['Idate'].year)), 
                     fill=True,fill_color=colormap(int(row['Idate'].year)),
		     fill_opacity=0.7,
                     popup=str(row['Idate'])))	  

   macarte.add_child(mc)
   htmlname = '../html/macarte_clusters.html'
   macarte.save(htmlname)


if Mapall:
   # the amout of PV is too high to render correctly so
   #for all NL plot data aggreagted per gemeente of year of installation.
   import folium
   import branca.colormap as bcm
   import matplotlib.mlab as mlab
   from mpl_toolkits.basemap import Basemap
   import geopandas

   coldict = df_PV.groupby('gemeente').count().huisnummer
   colormap=bcm.linear.Set3.scale(coldict.min(),coldict.max())

   def coldef(GM_NAAM):
    	try:
    	    colval = colormap(coldict.loc[GM_NAAM])
    	except:
    	    print ('no solar panels in:',GM_NAAM )
    	    colval = 'lightgrey'
   	# ok
    	return colval
   gem_shp = geopandas.read_file('/home/bigdata07/DataMatch/gem_2017.shp' )
   gem_shp.crs 
   gem_shp.index = gem_shp.GM_NAAM

   gem_shp = gem_shp[['GM_NAAM','geometry']]
   gem_ll = gem_shp.to_crs(epsg=4326)
   
   gem_json = gem_ll.to_json(na='drop')
   macarte = folium.Map(location=[gem_shp.crs['lat_0'],gem_shp.crs['lon_0']], zoom_start=8)
   folium.GeoJson(
          gem_json,
          style_function=lambda feature: {
        			'fillColor': coldef(feature['id']),
        			'color': 'black',
        			'weight': 1,
        			'dashArray': '5, 5',
        			'fillOpacity': 0.9,
				}
		  ).add_to(macarte)
   colormap.caption = 'aantal PV'
   macarte.add_child(colormap)   
   macarte.save('macarte_nederland.html')
   
   
 #shafile location=>  \\cbsp.nl\Productie\Secundair\GEODATA\Output\Cartografie check for provincie shapefile.

#endif
if Mapprov:

   import folium
   import branca.colormap as bcm
   from folium.plugins import MarkerCluster
   import geopandas
   import os.path
   print(folium.__version__)

   opath ='/home/bigdata07/DeepSol/html/'
   ipath ='/home/bigdata07/shpfile/'
   fname='pv_2018.shp'
   pv_shp = geopandas.read_file(os.path.join(ipath,fname))
   pv_shp.crs
   pv_shp.index = pv_shp.PV_NAAM
   pv_ll = pv_shp.to_crs(epsg=4326)

   map_loc = pv_ll.centroid

   colormap = bcm.linear.Set3.scale(2010,2020)
   for iprov in np.unique(df_PV.provincie):
       temp = df_PV[df_PV.provincie == iprov]
       macarte = folium.Map(location=[map_loc[iprov].coords[0][1],map_loc[iprov].coords[0][0]], zoom_start=10)
       mc = MarkerCluster()
       # mark each station as a point
       for index, row in temp.iterrows():
    	   try:
    	       mc.add_child(folium.CircleMarker([row['lat'], row['lon']],
    	    		    radius=5,
    	    		    color= colormap(int(row['Idate'].year)), 
    	    		    fill=True,fill_color=colormap(int(row['Idate'].year)),
	    		    fill_opacity=0.7,
    	    		    popup=str(row['Idate'])))
    	   except:
    	      print('Got an exception')
    	      print (row)
       macarte.add_child(mc) 
       colormap.caption = 'Installation Date'
       macarte.add_child(colormap)
       htmlname = 'macarte_'+ iprov+ '.html'
       macarte.save(os.path.join(opath,htmlname))

#iprov     
#eof


