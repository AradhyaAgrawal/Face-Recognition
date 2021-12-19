import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
import time
import glob

from PIL import Image

def main():
	imagePath = "img.jpg"
	
	img = cv2.imread(imagePath)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	generate_histogram(gray)
	
	cv2.imwrite("before.jpg", gray)

	gray = cv2.equalizeHist(gray)
	
	generate_histogram(gray)
	
	cv2.imwrite("after.jpg",gray)
	
	return 0

people = input("enter no. of people")
people = int(people)
numpy_image = []
name_array = []
for i in range(people):
    user_name =input("Enter your Name")
    name_array.append(user_name)
    print("Please capture 10 images with different movements")
    count = 1
    
            try:
                os.mkdir(dirName)
                print("") 
            except FileExistsError:
                print("")
            cv2.imwrite("faces/"+user_name+"/"+user_name+str(count)+".png",crop_face)

            cv2.imshow('frame',gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        count +=1
    cv2.destroyAllWindows()




numpy_image = []
files = glob.glob("faces/*/**.png", recursive=True)
for myFile in files:
    
    image = cv2.imread (myFile,0)
    numpy_image.append(image)

print('Numpy_image shape:', np.array(numpy_image).shape)




numpy_image = np.asarray(numpy_image)
scaled_images = []
for img in numpy_image:
    
    scale_percent = 30 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    scaled_images.append(resized)
plt.imshow(scaled_images[0])
plt.show()

flatten_image_array =[]
numpy_image = np.array(scaled_images)
for image in numpy_image:
    flatten_image_array.append(image.reshape(-1))


flatten_image_array = np.array(flatten_image_array)
flatten_image_array

import pickle
numpy_image = np.asarray(flatten_image_array)
example = numpy_image
pickle_out = open("dict.pickle","wb")
pickle.dump(example,pickle_out)
pickle_out.close()


import pickle
numpy_image = np.asarray(name_array)
example = numpy_image

pickle_out = open("dict1.pickle","wb")
pickle.dump(example,pickle_out)
pickle_out.close()


import pickle
numpy_image = np.asarray(people)
example = numpy_image
pickle_out = open("dict0.pickle","wb")
pickle.dump(example,pickle_out)
pickle_out.close()




import cv2

videoCaptureObject = cv2.VideoCapture(0)
while(True):
    ret,frame = videoCaptureObject.read()
    cv2.imshow('Capturing Video',frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        videoCaptureObject.release()
        cv2.destroyAllWindows()
