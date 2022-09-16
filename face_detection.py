from fileinput import filename
from torchvision.utils import save_image
from main import *
from torch import tensor

import cv2
import matplotlib.pyplot as plt
import numpy as np

#final filter

centroids_channel = [[tensor([[0.9140, 0.9176, 0.9140, 0.9168, 0.9209, 0.9166, 0.9130, 0.9167, 0.9126]],
       dtype=torch.float64)], [tensor([[0.6942, 0.6978, 0.6945, 0.6960, 0.7002, 0.6962, 0.6920, 0.6956, 0.6920]],
       dtype=torch.float64)], [tensor([[0.8402, 0.8469, 0.8415, 0.8505, 0.8588, 0.8518, 0.8406, 0.8469, 0.8414]],
       dtype=torch.float64)]]

icm_channel = [[tensor([[ 330.2361,    7.8850,  -93.2959,  -24.6380, -218.9388,  -92.3460,
          -97.1352,  -56.6551,  240.3955],
        [   7.8850,  315.2766,    4.6576, -251.3327,   71.0177, -260.7116,
          -55.0624,  261.3349,  -45.6933],
        [ -93.2959,    4.6576,  345.4090,  -96.6002, -228.8544,  -32.6630,
          244.8200,  -52.8484,  -95.8594],
        [ -24.6380, -251.3327,  -96.6002,  335.7118,  146.6682,  295.8012,
          -38.8262, -244.5342,  -71.4437],
        [-218.9388,   71.0177, -228.8544,  146.6682,  552.9655,  152.4601,
         -222.6862,   92.2754, -224.2504],
        [ -92.3460, -260.7116,  -32.6630,  295.8012,  152.4601,  359.4618,
          -71.0753, -254.3485,  -45.6170],
        [ -97.1352,  -55.0624,  244.8200,  -38.8262, -222.6862,  -71.0753,
          332.5986,   -8.3441,  -92.2295],
        [ -56.6551,  261.3349,  -52.8484, -244.5342,   92.2754, -254.3485,
           -8.3441,  318.0812,   -8.7996],
        [ 240.3955,  -45.6933,  -95.8594,  -71.4437, -224.2504,  -45.6170,
          -92.2295,   -8.7996,  332.9781]],
       dtype=torch.float64)], [tensor([[ 177.9739,   -6.9062, -113.6943,   25.5019,  -44.8342,   -5.9014,
         -115.3806,  -40.2389,  129.5516],
        [  -6.9062,   20.5046,   32.8672,   -1.1407,    7.7168,    0.2576,
            1.9009,  -12.1393,  -36.5366],
        [-113.6943,   32.8672,  114.5855,  -20.9770,   36.0429,    8.9931,
           62.2833,    5.5076, -118.8264],
        [  25.5019,   -1.1407,  -20.9770,   15.1367,   -3.2021,  -12.2663,
            3.7667,   -1.6459,    0.8734],
        [ -44.8342,    7.7168,   36.0429,   -3.2021,   14.3539,   -0.7639,
           34.2097,    8.0873,  -45.3450],
        [  -5.9014,    0.2576,    8.9931,  -12.2663,   -0.7639,   14.2452,
          -15.9859,   -0.3621,   18.1000],
        [-115.3806,    1.9009,   62.2833,    3.7667,   34.2097,  -15.9859,
          111.0012,   35.2421, -111.0146],
        [ -40.2389,  -12.1393,    5.5076,   -1.6459,    8.0873,   -0.3621,
           35.2421,   21.7794,  -10.2551],
        [ 129.5516,  -36.5366, -118.8264,    0.8734,  -45.3450,   18.1000,
         -111.0146,  -10.2551,  179.2536]],
       dtype=torch.float64)], [tensor([[ 56.5255,  15.2166, -26.4683,  -0.9177, -23.4540, -31.6453, -10.3455,
           4.5702,  23.0759],
        [ 15.2166,  14.2141,  14.2946, -14.1058, -20.6745, -14.5336,   4.4479,
           3.3763,   4.9855],
        [-26.4683,  14.2946,  58.3571, -31.6755, -23.4688,  -0.9361,  23.4149,
           5.0794, -11.2043],
        [ -0.9177, -14.1058, -31.6755,  46.8344,  41.8160,  12.1548,  -1.7403,
         -13.9923, -30.0119],
        [-23.4540, -20.6745, -23.4688,  41.8160,  59.4048,  42.4811, -23.3038,
         -20.5295, -22.7557],
        [-31.6453, -14.5336,  -0.9361,  12.1548,  42.4811,  48.7842, -31.5236,
         -14.4737,  -0.7098],
        [-10.3455,   4.4479,  23.4149,  -1.7403, -23.3038, -31.5236,  58.2167,
          15.2058, -27.8819],
        [  4.5702,   3.3763,   5.0794, -13.9923, -20.5295, -14.4737,  15.2058,
          14.1564,  13.9719],
        [ 23.0759,   4.9855, -11.2043, -30.0119, -22.7557,  -0.7098, -27.8819,
          13.9719,  58.2394]], dtype=torch.float64)]]



number_of_dimensions = 9
gamma = 1

source = "./dataset_test/test/001160.jpg"
filename = "001160"

img_org = Image.open(source)
original = cv2.imread(source)
convert_tensor = transforms.ToTensor()
img_org = convert_tensor(img_org)
result = calculate_final_result(icm_channel, centroids_channel, gamma, number_of_dimensions, img_org.reshape((1, 3, 218, 178)), 3, 1) # with label

first = "./dataset_train/first_output_train/" + filename + ".jpg"
save_image(result, first)

img = cv2.imread(first)
ret,thresh = cv2.threshold(img,200,255,cv2.THRESH_BINARY)

kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
opening = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
opening = cv2.convertScaleAbs(opening)


# noise = "./dataset_train/noise_removal_train/" + filename + ".jpg"
# cv2.imwrite(noise, opening)

# plt.imshow(opening)
# plt.show()

contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE,   cv2.CHAIN_APPROX_SIMPLE)

areas = []

centersX = []
centersY = []

for cnt in contours:

    areas.append(cv2.contourArea(cnt))

    M = cv2.moments(cnt)
    centersX.append(int(M["m10"] / M["m00"]))
    centersY.append(int(M["m01"] / M["m00"]))


full_areas = np.sum(areas)

acc_X = 0
acc_Y = 0

for i in range(len(areas)):

    acc_X += centersX[i] * (areas[i]/full_areas) 
    acc_Y += centersY[i] * (areas[i]/full_areas)



center_coordinates = (int(acc_X), int(acc_Y))
  
axesLength = (40, 70)
  
angle = 0
  
startAngle = 0
  
endAngle = 360
   
# Red color in BGR
color = (255, 0, 0)
   
# Line thickness of 5 px
thickness = 2


face = cv2.ellipse(original, center_coordinates, axesLength,
           angle, startAngle, endAngle, color, thickness)

face_detected = "./dataset_train/face_detected_train/" + filename + ".jpg"
# cv2.imwrite(face_detected, face)

plt.imshow(face)
plt.show()