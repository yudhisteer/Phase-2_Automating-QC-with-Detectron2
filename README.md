# Phase-2_Automating-QC-with-Detectron2

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
5. Build Detectron2 model
6. Train the model
7. Inference
8. Metrics

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
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()
```

![image](https://user-images.githubusercontent.com/59663734/138272628-dd69f5a7-b4f2-45c3-9d3a-fa2c15b96063.png)

From the Canny Edge Detector, we can clearly see the outline of our garment. The outline also clearly shows the contours which we can use for the measurements. The next step woud be to detect the corners or contours in our edges which will allow us to get the distance between two desired points.

### 2. Harris Corner Deteciton

The idea behind the Harris method is to detect points based on the intensity variation in a local neighborhood: a small region around the feature should show a large intensity change when compared with windows shifted in any direction.

```
img = cv2.imread(r"shirt.jpg")
img = cv2.resize(img, (0,0),None, 0.2, 0.2)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
```

![image](https://user-images.githubusercontent.com/59663734/138276853-2aa2b45f-bc9f-48ab-b4d1-ab110c894f35.png)

Even a close-up of a sleeve shows that not only the contours are detected. Wherever there is a stark contrast between two segments of the image the algo classifies it as a corner. This would be very inconvenient for our object measurement model as we would have a 
![image](https://user-images.githubusercontent.com/59663734/138276896-46cd44d3-bff2-4b09-a320-01a86edc905f.png)



### 3. Mouse Click Measurement

![image](https://user-images.githubusercontent.com/59663734/138279616-bd3b0eb7-3b8c-4926-a26c-27e42aea0a87.png)



### 4. Data Labeling

![image](https://user-images.githubusercontent.com/59663734/138276005-f09a4b4f-9e8c-4559-980a-7dd8d2c3ea61.png)


### 5. Build Detectron2 model

### 6. Train the model

### 7. Inference

### 8. Metrics


## Next Step

## Conclusion

## References

1. https://learnopencv.com/edge-detection-using-opencv/
