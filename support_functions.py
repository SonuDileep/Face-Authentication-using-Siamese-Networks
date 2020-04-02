import numpy as np
import cv2
import os
import random
np.random.seed(2)
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.display import clear_output
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import tensorflow as tf
import keras
from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import optimizers
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras.regularizers import l2


def from_camera(model,img1_path,cascade_path,display_image=False): # to verify an image stored in our database 
                                                                   # to an image from camera
    img1=cv2.imread(img1_path)
    face_cascade = cv2.CascadeClassifier(cascade_path)
    size=160
    img1 = cv2.imread(img1_path)
    if len(img1)==0:
        print("No image found in Path 1,Please check the filename")
        return 
    faces1 = face_cascade.detectMultiScale(img1, 1.05, 15)
    if len(faces1)==0:
        print("Could not find face in first image,Please try another image")
        return
    i=0
    for (x, y, w, h) in faces1:
        if i==0:
          img=img1[y:y+h,x:x+w]
          img1=cv2.resize(img,(size,size))
          i=i+1
    test1 =img1.reshape(1,size,size,3)
    test1=standardize_image(test1)
        
    vid = cv2.VideoCapture(-1)
    try:
        while(True):
            ret, frame = vid.read()
            print(ret)
            if ret==True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                clear_output(wait=True)
                faces2 = face_cascade.detectMultiScale(frame, 1.05, 15)
                i=0
#                 p.add_subplot(1,3, 2)
                plt.axis("off")
                plt.imshow(frame)
                plt.show()
                if len(faces2)!=0:
                    for (x, y, w, h) in faces2:
                        if i==0:
                          img=frame[y:y+h,x:x+w]
                          img2=cv2.resize(img,(size,size))
                          i=i+1
                          if display_image: # to output both images
                            p = plt.figure()
                            p.add_subplot(1,2, 1)
                            plt.axis("off")
                            plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
                            plt.title("database_image")
                            p.add_subplot(1,2, 2)
                            plt.axis("off")
                            plt.imshow(img2)
                            plt.title("Camera_face")
                            plt.show(block=True)   
                    test2 =img2.reshape(1,size,size,3)
                    test2=standardize_image(test2)
                    distance=model.predict([test1,test2])


                    print(distance)
                    if distance<1:
                        print("Similar Images")
                    if distance>1:
                        print("Dissimilar Images")
                    return distance
            else:
                return 0
    except KeyboardInterrupt:
        vid.release() 

    return 0





def output_model_1(model,img1_path,img2_path,show_image=True): # to test first model
    size=160

    face_cascade = cv2.CascadeClassifier('/home/sonu/Documents/Face_Ver_project/haarcascade_frontalface_default.xml')
    img1 = cv2.imread(img1_path)
    faces1 = face_cascade.detectMultiScale(img1)
    i=0
    for (x, y1, w, h) in faces1:
        if i==0:
            img=img1[y1:y1+h,x:x+w]
            img1=cv2.resize(img,(size,size))
            i=i+1
    test1=standardize_image(img1)
    test1 =test1.reshape(1,size,size,3)


    img2 = cv2.imread(img2_path)
    faces2 = face_cascade.detectMultiScale(img2)
    i=0
    for (x, y1, w, h) in faces2:
        if i==0:
            img=img2[y1:y1+h,x:x+w]
            img2=cv2.resize(img,(size,size))
            i=i+1
    test2=standardize_image(img2)        
    test2 =test2.reshape(1,size,size,3)
    if show_image: # to output both images
        p = plt.figure()
        p.add_subplot(1,2, 1)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title("image1")
        p.add_subplot(1,2, 2)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.title("image2")
        plt.show(block=True)  
    p = model.predict([test1,test2])
    if p<0.6:
        print("Similar Images")
        print("Dissimilarity score = ",p[0])
    else:
        print("Dissimilar Images")
        print("Dissimilarity score = ",p[0])
    return p



def read_image2(path,num,number_images): # function to read face12 dataset
    import random  
    random.seed(30)
    size=160
    face_cascade = cv2.CascadeClassifier('/home/sonu/Documents/Face_Ver_project/haarcascade_frontalface_default.xml')

    folder_name = [x[0] for x in os.walk(path)]
    full_loc=np.empty((num,15), dtype='object')
    g=folder_name[0]
    del folder_name[0]
    for i in range(len(folder_name)):
        inside_folder=[]
        inside_folder = os.listdir(folder_name[i])
        for j in range(0,len(inside_folder)):
            a,b = inside_folder[j].split('.', 1)
            inside_folder[j]=(folder_name[i] + '/' + inside_folder[j])
            full_loc[i][j]=inside_folder[j] 
            
    img_array1=[]
    img_array2=[]
    Y=[]
    size=160
    n=number_images # number of samples you need
    avoid_similar=[] # to avoid repetition of same dataset
    t=0
    for i in range(0, n):
        s=random.choice([0,1])
        if s==1: # similar image
            img_number=random.randrange(0,num)
            img1 = []
            img2 = []
            name1 = full_loc[img_number][random.randrange(0,15)]
            name2 = full_loc[img_number][random.randrange(0,15)]
            if (name1,name2) not in avoid_similar:
                img1 = cv2.imread(name1)
                img2 = cv2.imread(name2)
                
                faces1 = face_cascade.detectMultiScale(img1)
                if len(faces1)==0:
                    img1=cv2.resize(img1,(size,size))
                else:
                    i=0
                    for (x, y1, w, h) in faces1:
                        if i==0:
                            img=img1[y1:y1+h,x:x+w]
                            img1=cv2.resize(img,(size,size))
                            i=i+1

                faces2 = face_cascade.detectMultiScale(img2)
                if len(faces2)==0:
                    img2=cv2.resize(img2,(size,size))
                else:
                    i=0
                    for (x, y1, w, h) in faces2:
                        if i==0:
                            img=img2[y1:y1+h,x:x+w]
                            img2=cv2.resize(img,(size,size))
                            i=i+1
                img1=standardize_image(img1)
                img2=standardize_image(img2)
                img_array1.append(img1)
                img_array2.append(img2)           
                Y.append('1')
                avoid_similar.append((name1,name2))
                t=t+1
            else:
                i=i-1
            
        if s==0: # dissimilar image
            img_number1 = random.randrange(0,num-1)
            img_number2 = random.randrange(0,num-1)
            if img_number1 == img_number2:
                if img_number1 == 0:
                    img_number1 = 1
                if img_number1 == num-1:
                    img_number1 = num-1
            img1 = []
            img2 = []
            name1 = full_loc[img_number1][random.randrange(0,15)]
            name2 = full_loc[img_number2][random.randrange(0,15)]
            if (name1,name2) not in avoid_similar:
                img1 = cv2.imread(name1)
                img2 = cv2.imread(name2)
                
                faces1 = face_cascade.detectMultiScale(img1)
                if len(faces1)==0:
                    img1=cv2.resize(img1,(size,size))
                else:
                    i=0
                    for (x, y1, w, h) in faces1:
                        if i==0:
                            img=img1[y1:y1+h,x:x+w]
                            img1=cv2.resize(img,(size,size))
                            i=i+1

                faces2 = face_cascade.detectMultiScale(img2)
                if len(faces2)==0:
                    img2=cv2.resize(img2,(size,size))
                else:
                    i=0
                    for (x, y1, w, h) in faces2:
                        if i==0:
                            img=img2[y1:y1+h,x:x+w]
                            img2=cv2.resize(img,(size,size))
                            i=i+1
                img1=standardize_image(img1)
                img2=standardize_image(img2)
                img_array1.append(img1)
                img_array2.append(img2)   
                
                Y.append('0')
                avoid_similar.append((name1,name2))
                t=t+1
            else:
                i=i-1
                
    data1 = np.asarray(img_array1)
    data2 = np.asarray(img_array2)
    data1=data1.reshape(t,size,size,3)
    data2=data2.reshape(t,size,size,3)
    y = np.asarray(Y)
    y = y.reshape(t,1)
#     data=np.hstack((data1,data2))
#     data = data/255.0    
    return data1,data2,y



def standardize_image(img): # Function to standardize the input images
  img=img/255.0 
  mean=np.mean(img,keepdims=True)
  std=np.std(img,keepdims=True)
  return (img-mean)/std

def read_image(path,num,number_images):# Function to read faces95 dataset for training and validating
    import random                                  
    folder_name = [x[0] for x in os.walk(path)]
    full_loc=np.empty((num,20), dtype='object')
    g=folder_name[0]
    del folder_name[0]
    for i in range(len(folder_name)):
        inside_folder=[]
        inside_folder = os.listdir(folder_name[i])
        for j in range(0,len(inside_folder)):
            a,b = inside_folder[j].split('.', 1)
            inside_folder[j]=(g + '/' + a +'/' + inside_folder[j])
            full_loc[i][j]=inside_folder[j] 
            
    img_array1=[]
    img_array2=[]
    Y=[]
    size=160
    n=number_images # number of samples you need
    avoid_similar=[] # to avoid repetition of same dataset
    t=0
    for i in range(0, n):
        s=random.choice([0,1])
        if s==1: # similar image
            img_number=random.randrange(0,num)
            img1 = []
            img2 = []
            name1 = full_loc[img_number][random.randrange(0,20)]
            name2 = full_loc[img_number][random.randrange(0,20)]
            if (name1,name2) not in avoid_similar:
                img1 = cv2.imread(name1)
                img2 = cv2.imread(name2)
                img1=standardize_image(img1)
                img2=standardize_image(img2)
                img_resized1 = cv2.resize(img1, (size,size))
                img_resized2 = cv2.resize(img2, (size,size))
                img_array1.append(img_resized1)
                img_array2.append(img_resized2)           
                Y.append('1')
                avoid_similar.append((name1,name2))
                t=t+1
            else:
                i=i-1
            
        if s==0: # dissimilar image
            img_number1 = random.randrange(0,num-1)
            img_number2 = random.randrange(0,num-1)
            if img_number1 == img_number2:
                if img_number1 == 0:
                    img_number1 = 1
                if img_number1 == num-1:
                    img_number1 = num-1
            img1 = []
            img2 = []
            name1 = full_loc[img_number1][random.randrange(0,20)]
            name2 = full_loc[img_number2][random.randrange(0,20)]
            if (name1,name2) not in avoid_similar:
                img1 = cv2.imread(name1)
                img2 = cv2.imread(name2)
                img1=standardize_image(img1)
                img2=standardize_image(img2)
                img_resized1 = cv2.resize(img1, (size,size))
                img_resized2 = cv2.resize(img2, (size,size))
                img_array1.append(img_resized1)
                img_array2.append(img_resized2)
                Y.append('0')
                avoid_similar.append((name1,name2))
                t=t+1
            else:
                i=i-1
                
    data1 = np.asarray(img_array1)
    data2 = np.asarray(img_array2)
    data1=data1.reshape(t,size,size,3)
    data2=data2.reshape(t,size,size,3)
    y = np.asarray(Y)
    y = y.reshape(t,1)   
    return data1,data2,y


