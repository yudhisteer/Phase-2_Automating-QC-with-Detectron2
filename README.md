# Phase-2: Automating QC with Detectron2

## Abstract

## How to measure?

Different clients require different types of measurement. For the purpose of this project, I chose to automate the quality control process for T-shirts only. Some clients require staright line measurement between 2 points and others require the length of a curvature or even some require the length of two legs of a 90 degrees triangle of a curve part. Below is a How to measure? sheet in a tech pack which shows the process of taking measurements:

![image](https://user-images.githubusercontent.com/59663734/138277860-7666599e-9d3e-4e52-81ec-ac22a76b9622.png)

### AQL
'AQL' stands for **'Acceptance Quality Limit'**, and is defined as the “quality level that is the worst tolerable” in ISO 2859-1. It represents the maximum number of defective units, beyond which a batch is rejected. Based on the sampling data, the customer can make an informed decision to accept or reject the lot.

An AQL result of 1.5 accepts the statistical probability that there are less than 1.5% of the products with defects in the batch. An AQL of 0.65 assumes a more stringent quality acceptance level. Below is a table of the different AQL required by the clients:

| Client | AQL |
| :---: | :---: |
| Adidas | 1.0 |
| LaCoste | 1.0 |
| ASOS | 2.5 |
| WoolWorths | 1.0 |
| Puma | 1.0 |
| Cape Union Mart | 2.5 |


## Plan of Action

1. Canny Edge Deteciton
2. Harris Corner Deteciton
3. Mouse Click Measurement
4. Data Labeling
5. JSON to COCO
6. Build Detectron2 model
7. Train the model
8. Inference
9. Metrics

Before diving straight into an AI model, I wanted to explore some image processing techniques that would allow me to pick the best model for object measurements. I started with the simplest of all: Canny Edge Detection and then moved on to Corner Detection.

### 1. Canny Edge Detection

Edge detection is an image-processing technique, which is used to identify the boundaries (edges) of objects, or regions within an image. It is identified by sudden changes in pixel intensity. Canny Edge Detection algorithm consists of 4 stages which includes:

**1. Noise Reduction**: We need to eliminate the noise in the image if we don't want unnecessary edges. A Gaussian Blue Filter is used to eliminate noise that could lead to unnecessary detail.

**2. Calculating Intensity Gradient of the Image**: Once the image has been smoothed (blurred), it is filtered with a Sobel kernel, both horizontally and vertically. The results are then used to calculate the intensity gradient and the direction for each pixel.

**3. Suppression of False Edges**: We use a technique called non-maximum suppression of edges to filter out unwanted pixels (which may not actually constitute an edge). Each pixel is compared to its neighboring pixels and if its gradient magnitude is greater than the neighboring one then it is left unchanged else it is set to zero.

**4. Hysteresis Thresholding**: The gradient magnitudes are compared with two extremes thresholds. If the gradient magnitude is above the larger threshold then it is marked as "strong edges". If it is lower than the lower threshold then the pixels are excluded. If they are in between the thresholds then they are marked as "weak edges".

Fortunately for us, the  Canny() function implements all the methodology described above.

We start by importing the necessary dependencies:

```
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
```


```
img = cv.imread('polo.jpg',0)
img_blur = cv.GaussianBlur(img,(3,3), sigmaX=0, sigmaY=0)
edges = cv.Canny(img_blur,100,200)
plt.subplot(121),plt.imshow(img_blur,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
```

![image](https://user-images.githubusercontent.com/59663734/138272628-dd69f5a7-b4f2-45c3-9d3a-fa2c15b96063.png)

From the Canny Edge Detector, we can clearly see the outline of our garment. The outline also clearly shows the contours which we can use for the measurements. The next step woud be to detect the corners or contours in our edges which will allow us to get the distance between two desired points.

### 2. Harris Corner Deteciton

Harris method is to detect points based on the intensity variation in a local neighborhood. It is a mathematical way to show which windows shows a large variations when moved in any direction. 

Commonly, Harris corner detector algorithm can be divided into five steps.

1. Color to grayscale
2. Spatial derivative calculation
3. Structure tensor setup
4. Harris response calculation
5. Non-maximum suppression

```
img = cv2.imread(r"shirt.jpg")
img = cv2.resize(img, (0,0),None, 0.2, 0.2)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
# Dilate corner image to enhance corner points
dst = cv2.dilate(dst,None)
# This value vary depending on the image and how many corners you want to detect
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
```
Unfortunately with the Harris Corner Detector, we get more points besides the corners. In some parts of the pictures, the corners are not even detected. The quality of the image and the background also is a factor to be taken into consideration. However, our measurements depend on the distance between two points and it will be hard for us to decipher which points is the corner points among all these points.

![image](https://user-images.githubusercontent.com/59663734/138276853-2aa2b45f-bc9f-48ab-b4d1-ab110c894f35.png)

Even a close-up of a sleeve shows that not only the contours are detected. Wherever there is a stark contrast between two segments of the image, the algorithm classifies it as a corner. This would be very inconvenient for our object measurement model as we would have a series of points and not 2 points signifying two corners. 

![image](https://user-images.githubusercontent.com/59663734/138276896-46cd44d3-bff2-4b09-a320-01a86edc905f.png)

From the two image processing techniques - Canny Edge Detection and Harris Corner Detection - we can conclude that it will not be reliable for us to get the measurmeents between two keypoints. We need a smarter model that will be able to pinpoint the exact location of the particular areas desired with a high degree of accuracy. Before going into an AI model, I wanted to simulate the pinpoint system. I started with a semi-automatic solution whereby the user would select the two points he would need to measure and the exact distance of the garment will be calculated.

### 3. Mouse Click Measurement

If using the Canny Edge Detection and Harris Corner Detection algorithms have not been promising, I decided to make the user pinpoints the parts of the garments he would want to measure and the distance will be calculated automatically. This would still greatly optimize the Quality 'Control process as instead of the QC people standing with a tape measure and taking on average 15 measurements, now he would just need to pinpoint using his mouse the different parts of the garment.

We start by crating two lists: ```xcoor``` and ```ycoor```. they both have an initial value of zero. When the user will pinpoint on the image, the x and y values of that point will be stored in the lists created.

```
index = 1
xcoor = [0]
ycoor = [0]
```

We have a function called ```click_event``` which checks for left mouse clicks from her user. If so, it creates a small green dot on the image using ```cv2.circle(img,(x,y), 3, (0, 255, 0), -1)``` at the x and y coordinates it has been pointed. 

When pinpointing two points on the image, we have 2 x coordinates: ```x1``` and ```x2``` and two y coordinates ```y1``` and ```y2```. We select them by their indexes and append them into the initial lists created.

The Euclidean Distance between the two points is calculated using ```dis = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5```. This is the distance in pixel. If we need to get the actual measurement of a part of the garment, we need to convert our pixel values into mm or cm. A scaling factor of 0.15 is used for the conversion: ```Actual_dis = round(dis * 0.15, 2)```

The actual distance need to be saved in a nexcel sheet. The google sheet API has been used to directly populate a template in real-time using ```Actual_dis = round(dis * 0.15, 2)```.


```
def click_event(event, x, y, flags, params):
    # checking for left mouse clicks

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y), 3, (0, 255, 0), -1)

        # displaying the coordinates
        # on the Shell
        print("------------------------")
        print("last x-y: ", x, ' ', y)
        xcoor.append(x)
        ycoor.append(y)

        print("xcoor = "+str(xcoor))
        print("ycoor = " + str(ycoor))
        x1 = xcoor[-2]
        x2 = xcoor[-1]
        y1 = ycoor[-2]
        y2 = ycoor[-1]

        print("x1 = "+str(x1))
        print("y1 = "+str(y1))
        print("x2 = " + str(x2))
        print("y2 = "+str(y2))

        dis = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        dis = abs(dis)
        #print(dis)
        print(" ")
        print("Pixel Distance = " + str(dis) + " px")
        print(" ")
        Actual_dis = round(dis * 0.15, 2)
        print("Actual Length = "+ str(Actual_dis)+ " cm")
        print(" ")

        # write in cells#
        global index
        index +=1
        print("index = "+str(index))
        if index ==2:
            print("Ignoring first input")
        else:
            sheet.update_cell(index-1, 2, Actual_dis)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    0.5, (255, 50, 0), 2)
        cv2.imshow('image', img)
```

With the x and y coordinates of the points stored in a list, the distance was calculated using the euclidean distance formula: d = √[(x2 – x1)^2 + (y2 – y1)^2]

![image](https://user-images.githubusercontent.com/59663734/138279616-bd3b0eb7-3b8c-4926-a26c-27e42aea0a87.png)

The manual measurement system using OpenCV would greatly optimize the QC process and reduce the measuring time of the garments. We tested the model and indeed it was moreefficient than the DeepSpeech solution. However, we wanted a fully end-to-end automated system. We need a system with very little to no human intervention and with a high degree of accuracy(tolerance of ± 10 mm) for taking the measurements.


### 4. Data Labeling

We will need to build a Keypoint Detection algorithm hence, we start by collecting data - images of t-shirts - scrapped from the Shein website using an RPA. To label the images, I made use of **label-studio** and used the script below to generate the labels that would be used.

```
<View>
	<Image name="image" value="$image" zoom="true" zoomControl="true"/>
	<KeyPointLabels name="keypoints" toName="image"
		strokewidth="2" opacity="1" >
		<Label value="left_sleeve_1" background="red"/>
		<Label value="left_sleeve_2" background="yellow"/>
		<Label value="right_sleeve_1" background="pink"/>
		<Label value="right_sleeve_2" background="blue"/>
	</KeyPointLabels>
	<RectangleLabels name="bbox" toName="image">
		<Label value="Shirt" background="green"/>
	</RectangleLabels>
</View>
```
For the sake of building a POC(Proof of Concept), I decided to build a keypoint model that would be able to detect the left and right sleeves of the garment. Upon approval by the board at RT Knits, we will then tackle the other measurements.

The label values are: ```left_sleeve_1, left_sleeve_2, right_sleeve_1, right_sleeve_2```. We will also need a bounding box ```Shirt``` that will be used to detect the position of the shirt in the image. 

![image](https://user-images.githubusercontent.com/59663734/138276005-f09a4b4f-9e8c-4559-980a-7dd8d2c3ea61.png)

About 180 pictures were downloaded and labelled. 20% of them were transferred in the ```test``` folder and the rest in the ```train``` folder. After labelling, the downloaded format of the data is in ```json```. However, in order to train our model we will need to convert our ```json``` file into ```COCO``` format. 

### 5. JSON to COCO

We start by importing the necessary libraries:

```
import json
import itertools
import cv2
import os
```

We will need the path of our ```train``` folder. I also created two empty lists: ```annotations``` and ```images``` that will be used to iterate inside a ```for``` loop to get the image id and the value of the coordinates of the keypoints and bounding box. 

```
DATA_DIR = ROOT_DIR+'train' #train images folder
annotations = []
images = []
```
The script below does the conversion into ```COCO``` format:

```
with open(ROOT_DIR+'labels/data.json') as f: #data.json is the output of label-studio
    d = json.load(f)
    for i, obj in enumerate(d):
        filename = obj['file_upload'].split('-', 1)[1]
        if not os.path.isfile(os.path.join(DATA_DIR, filename)):
            continue
        im = cv2.imread(os.path.join(DATA_DIR, filename))
        height, width, channels = im.shape
        
        image = {'id': i, 'file_name': filename, 'height': height, 'width': width}
        
        annotation = {'id':i, 'image_id':i, 'num_keypoints': 4, "category_id": 1}
        keypoints = {'left_sleeve_1': [],'left_sleeve_2': [],'right_sleeve_1': [],'right_sleeve_2': []} #keypoints
        for result in obj['annotations'][0]['result']:
            value = result['value']
            if result['type'] =='rectanglelabels':
                annotation['bbox'] = [round(value['x']/ 100.0 * width), round(value['y']/ 100.0 * height), round(value['width']/ 100.0 * width), round(value['height']/ 100.0 * height)]
            else:
                keypoints[value['keypointlabels'][0]] = [round(value['x']/ 100.0 * width), round(value['y']/ 100.0 * height), 2]
        annotation['keypoints'] = list(itertools.chain(*keypoints.values()))
        annotations.append(annotation)
        images.append(image)
        
categories = [{'id': 1, 'name': 'shirt', 'keypoints': ['left_sleeve_1', 'left_sleeve_2', 'right_sleeve_1', 'right_sleeve_2']}] # keypoints
        
final_annotations = {'images': images, 'annotations': annotations, 'categories':categories}
with open(ROOT_DIR+'labels/data_coco.json', 'w') as f: #data_coco.json is converted labels
    json.dump(final_annotations, f)
```

We save our converted labels into ```data_coco.json```.

### 6. Build Detectron2 model

(What is detectron2?)

We start by important some libraries and detectron2 utilities:

```
import numpy as np
import matplotlib.pyplot as plt
# import some common libraries
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger

from detectron2.data import (MetadataCatalog,
                             build_detection_train_loader,
                             build_detection_test_loader,
                             DatasetMapper, transforms as T,
                             detection_utils as utils)
setup_logger()
```


### 7. Train the model

### 8. Inference

### 9. Metrics


## Next Step

## Conclusion

## References

1. https://learnopencv.com/edge-detection-using-opencv/
2. https://medium.com/data-breach/introduction-to-harris-corner-detector-32a88850b3f6
