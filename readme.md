

# Google Earth engine for deep learning labelling 

- Modou MBAYE

Google Earth Engine is a powerful cloud-based platform for analyzing and visualizing geospatial data. It provides a vast amount of satellite imagery and other geospatial data that can be used for various applications, including generating label data for deep learning models like U-Net.

To generate label data using Google Earth Engine for a U-Net deep learning model, you can use the following code in the case of crop mapping .  



```javascript

Map.setOptions('SATELLITE')
var regions
Map.centerObject(regions,12)
var now=ee.Date(Date.now())

var S2=ee.ImageCollection('COPERNICUS/S2')
        .filterDate("2024-01-01","2024-01-30")
        .filterBounds(regions)
        .select('B2','B3','B4','B8','B11','B12')
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE','less_than',1)
        .map(function(image){ 
          var ndvi=image.normalizedDifference(['B8','B4']).rename('NDVI')
          var hue=image.expression('2*(b-g-r)/(30.5*(g-r))',
                  {r:image.select('B4'),
                  g:image.select('B3'),
                  b:image.select('B2')}).atan().rename('HUE')
          return image.addBands([ndvi,hue])
          
        })



print(S2.size())
Map.addLayer(S2, {min:[], max:[], bands:["B11","B8","B3"]},'false image')

var mosaic = S2.max().clip(regions)
print(mosaic)


var soil=mosaic.select("HUE").gt(0)
var masks=mosaic.select("HUE").lt(0).and(mosaic.select("NDVI").gt(0)).selfMask()
var crop= mosaic.mask(masks)
Map.addLayer(masks,{},'mask_soil')

var shapefiles=masks.reduceToVectors({
  geometry:regions,
  scale:10,
  geometryType:'polygon', //or "bb" tog generate box
  maxPixels:1e13,
 
})
var shapefiles=shapefiles.filter(ee.Filter.eq('label',1))
shapefiles=shapefiles.map(function(shape){return shape.set('area',shape.area(1))})
print("statistics ",shapefiles.aggregate_stats('area'))
shapefiles=shapefiles.filter(ee.Filter.gt('area',3000))


Export.table.toDrive(shapefiles,'vector_DeepLearning_Senegal_Nord', "labels_Data")
Export.image.toDrive({
image:mosaic.toFloat(), 
region:validations, 
folder:'Senegal_DeepLearning', 
description:'S2_bands',
scale:10, 
maxPixels:1e9})

```


Crop mapping using deep learning has become a transformative approach in agriculture, allowing for precise identification, classification, and monitoring of various crop types across extensive areas. Deep learning methods, particularly convolutional neural networks (CNNs), have significantly advanced this field by leveraging their capability to recognize complex patterns within high-dimensional data. This document provides an overview of the essential elements of deep learning in crop mapping, emphasizing its benefits in terms of accuracy, scalability, adaptability, and its broad impact on agricultural methodologies.


![Crop mapping](https://github.com/mmbaye/U_NET_GoogleEarthEngine/blob/main/Figs/crop_mapping.png)

![Link to the original paper of Unet](https://arxiv.org/abs/1505.04597)

![Unet Architecture](https://github.com/mmbaye/U_NET_GoogleEarthEngine/blob/main/Figs/U-Net.png)





