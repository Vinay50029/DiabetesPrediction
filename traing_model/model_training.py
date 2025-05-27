from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import os
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model
from keras.models import model_from_json
import pickle
from sklearn.model_selection import train_test_split
from keras.applications import DenseNet121
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from keras.applications import ResNet101
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

main = tkinter.Tk()
main.title("Diabetic Retinopathy Detection using ResNet101 and DenseNet121") 
main.geometry("1300x1200")

global filename
global densenet, resnet, X_train, X_test, y_train, y_test
global accuracy, precision, recall, fscore
labels = ['Mild DR', 'Moderate DR', 'No DR', 'Proliferative DR', 'Severe DR']

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index        

def uploadDataset(): 
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def preprocessDataset():
    global X, Y, X_train, X_test, y_train, y_test, filename
    text.delete('1.0', END)
    if os.path.exists('model/X.txt.npy'):
        X = np.load('model/X.txt.npy')
        Y = np.load('model/Y.txt.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (32,32))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(32,32,3)
                    X.append(im2arr)
                    label = getID(name)
                    Y.append(label)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    X = X.astype('float32')
    X = X/255
    test = X[3]
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)  
    X = X[indices]
    Y = Y[indices]
    Y = to_categorical(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test
    cv2.imshow("Processed Image", cv2.resize(test, (128, 128)))
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n")
    text.insert(END,"Class Labels found in dataset : "+str(labels)+"\n\n")
    text.insert(END,"Dataset Train & Test Split Details\n\n")
    text.insert(END,"80% dataset images using to train Resnet & Densenet : "+str(X_train.shape[0])+"\n")
    text.insert(END,"20% dataset images using to test Resnet & Densenet  : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
    cv2.waitKey(0)

def calculateMetrics(algorithm, test_y, predict):
    p = precision_score(test_y, predict,average='macro') * 100
    r = recall_score(test_y, predict,average='macro') * 100
    f = f1_score(test_y, predict,average='macro') * 100
    a = accuracy_score(test_y,predict)*100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy   : "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FSCORE    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(test_y, predict)
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()        

def runResent():
    global accuracy, precision, recall, fscore, resnet
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    accuracy = []
    precision = []
    recall = []
    fscore = []
    resnet = ResNet101(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in resnet.layers:
        layer.trainable = False
    resnet = Sequential()
    resnet.add(Convolution2D(32, (1 , 1), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    resnet.add(MaxPooling2D(pool_size = (1, 1)))
    resnet.add(Convolution2D(32, (1, 1), activation = 'relu'))
    resnet.add(MaxPooling2D(pool_size = (1, 1)))
    resnet.add(Flatten())
    resnet.add(Dense(units = 256, activation = 'relu'))
    resnet.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    resnet.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/resnet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/resnet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = resnet.fit(X_train, y_train, batch_size = 64, epochs = 100, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/resnet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        resnet.load_weights("model/resnet_weights.hdf5")
    predict = resnet.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("ResNet101", testY, predict)
    
def runDensenet():
    global X_train, X_test, y_train, y_test, densenet
    densenet = DenseNet121(include_top=False, weights='imagenet', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    for layer in densenet.layers:
        layer.trainable = False
    densenet = Sequential()
    densenet.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
    densenet.add(MaxPooling2D(pool_size = (2, 2)))
    densenet.add(Convolution2D(32, (3, 3), activation = 'relu'))
    densenet.add(MaxPooling2D(pool_size = (2, 2)))
    densenet.add(Flatten())
    densenet.add(Dense(units = 256, activation = 'relu'))
    densenet.add(Dense(units = y_train.shape[1], activation = 'softmax'))
    densenet.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    if os.path.exists("model/densenet_weights.hdf5") == False:
        model_check_point = ModelCheckpoint(filepath='model/densenet_weights.hdf5', verbose = 1, save_best_only = True)
        hist = densenet.fit(X_train, y_train, batch_size = 64, epochs = 100, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
        f = open('model/densenet_history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()    
    else:
        densenet.load_weights("model/densenet_weights.hdf5")
    predict = densenet.predict(X_test)
    predict = np.argmax(predict, axis=1)
    testY = np.argmax(y_test, axis=1)
    calculateMetrics("DenseNet121", testY, predict)

def graph():
    df = pd.DataFrame([['ResNet101','Precision',precision[0]],['ResNet101','Recall',recall[0]],['ResNet101','F1 Score',fscore[0]],['ResNet101','Accuracy',accuracy[0]],
                       ['DenseNet121','Precision',precision[1]],['DenseNet121','Recall',recall[1]],['DenseNet121','F1 Score',fscore[1]],['DenseNet121','Accuracy',accuracy[1]],
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()


def predictDisease(image_path, side):
    image = cv2.imread(image_path)
    img = cv2.resize(image, (32, 32))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = densenet.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (700,400))
    cv2.putText(img, side+' Retinopathy Predicted as : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
    return img

def predict():
    global densenet
    filename = filedialog.askdirectory(initialdir="testImages")
    left = predictDisease(filename+"/left.png", "Left Eye")
    right = predictDisease(filename+"/right.png", "Right Eye")
    cv2.imshow('Left Eye', left)
    cv2.imshow('Right Eye', right)
    cv2.waitKey(0)   


font = ('times', 16, 'bold')
title = Label(main, text='Diabetic Retinopathy Detection using ResNet101 and DenseNet121')
title.config(bg='firebrick4', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Diabetic Retinopathy Dataset", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=preprocessDataset, bg='#ffb3fe')
processButton.place(x=440,y=550)
processButton.config(font=font1) 

resnetButton1 = Button(main, text="Run ResNet101 Algorithm", command=runResent, bg='#ffb3fe')
resnetButton1.place(x=670,y=550)
resnetButton1.config(font=font1) 

densenetButton = Button(main, text="Run DenseNet121 Algorithm", command=runDensenet, bg='#ffb3fe')
densenetButton.place(x=50,y=600)
densenetButton.config(font=font1) 

graphButton = Button(main, text="Comparison Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=440,y=600)
graphButton.config(font=font1) 

predictButton = Button(main, text="Retinopathy Detection from Test Images", command=predict, bg='#ffb3fe')
predictButton.place(x=670,y=600)
predictButton.config(font=font1) 

main.config(bg='LightSalmon3')
main.mainloop()
