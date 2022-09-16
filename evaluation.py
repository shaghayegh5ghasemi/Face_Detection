
# calculate the center of the oval for each image

from email.mime import image
from turtle import right
import cv2
import glob
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle


def find_center(opening, i):

    imgray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE,   cv2.CHAIN_APPROX_SIMPLE)

    areas = []

    centersX = []
    centersY = []

    for cnt in contours:

        areas.append(cv2.contourArea(cnt))

        M = cv2.moments(cnt)
        if(M["m00"] == 0):
            print(i)
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
    
    return center_coordinates, axesLength

def get_labels(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    labels = []
    for l in range(2, len(lines)):
        temp = lines[l].split()
        temp.pop(0) # remove the first element of list
        res = [eval(i) for i in temp] # convert str to int
        labels.append(res)
    
    return labels

def checkpoint( center, axes_length, x, y):
    h, k = center
    a, b = axes_length
    # center = (h, k), axes_length = (a, b), point = (x, y) 
    # checking the equation of
    # ellipse with the given point
    p = ((math.pow((x - h), 2) // math.pow(a, 2)) +
         (math.pow((y - k), 2) // math.pow(b, 2)))
 
    return p

def compare(center, length, label):
    # lefteye_x lefteye_y righteye_x righteye_y nose_x nose_y leftmouth_x leftmouth_y rightmouth_x rightmouth_y
    left_eye = checkpoint(center, length, label[0], label[1])
    right_eye = checkpoint(center, length, label[2], label[3])
    nose = checkpoint(center, length, label[4], label[5])
    left_mouth = checkpoint(center, length, label[6], label[7])
    right_mouth = checkpoint(center, length, label[8], label[9])

    res = 5*[0]
    
    if left_eye < 1:
        res[0] = 1
    if right_eye < 1:
        res[1] = 1
    if nose < 1:
        res[2] = 1
    if left_mouth < 1:
        res[3] = 1
    if right_mouth < 1:
        res[4] = 1
    
    return res
       

images = [cv2.imread(file) for file in glob.glob('./dataset_test/noise_removal_test/*.jpg')]
labels = get_labels('./dataset_test/list_landmarks_align_celeba_test.txt')


left_eye = []
right_eye = []
nose = []
right_mouth = []
left_mouth = []
mouth = []

for i in range(len(images)):
    img_info = find_center(images[i], i)
    cmp = compare(img_info[0], img_info[1], labels[i])
    
    left_eye.append(cmp[0])
    right_eye.append(cmp[1])
    nose.append(cmp[2])
    right_mouth.append(cmp[3])
    left_mouth.append(cmp[4])

    if cmp[3] == 1 and cmp[4] == 1:
        mouth.append(1)
    else:
        mouth.append(0)


# all = []

# for i in range(len(images)):
#     img_info = find_center(images[i], i)
#     cmp = compare(img_info[0], img_info[1], labels[i])

#     sum_cmp = sum(cmp)
#     if sum_cmp > 2:
#         all.append(1)
#     else:
#         all.append(0)
    

# x_axis = [*range(1, len(images) + 1, 1)]
# y_axis = all
# plt.plot(x_axis, y_axis)
# plt.xlabel('test set') #x label
# plt.ylabel("detect eye, nose, mouth") #y label
# plt.show()

# x_axis = [*range(1, len(images) + 1, 1)]
# y_axis = right_eye
# plt.plot(x_axis, y_axis)
# plt.xlabel('test set') #x label
# plt.ylabel('right eye') #y label
# plt.show()

nose[44] = 0
nose[53] = 0
x_axis = [*range(1, len(images) + 1, 1)]
y_axis = nose
plt.plot(x_axis, y_axis)
plt.xlabel('test set') #x label
plt.ylabel('nose') #y label
plt.show()

# x_axis = [*range(1, len(images) + 1, 1)]
# y_axis = left_mouth
# plt.plot(x_axis, y_axis)
# plt.xlabel('test set') #x label
# plt.ylabel('left mouth') #y label
# plt.show()

# x_axis = [*range(1, len(images) + 1, 1)]
# y_axis = right_mouth
# plt.plot(x_axis, y_axis)
# plt.xlabel('test set') #x label
# plt.ylabel("right mouth") #y label
# plt.show()

# x_axis = [*range(1, len(images) + 1, 1)]
# y_axis = mouth
# plt.plot(x_axis, y_axis)
# plt.xlabel('test set') #x label
# plt.ylabel("mouth") #y label
# plt.show()