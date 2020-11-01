import cv2
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image

import PIL.ImageOps
import os,ssl,time

X = np.load('image.npz')['arr_0']
Y = pd.read_csv('labels.csv')['labels']

print(pd.Series(Y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
n_classes = len(classes)
print(n_classes)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=9, train_size=3500, test_size=500)

X_train_scale = X_train/255.0
X_test_scale = X_test/255.0

clf = LogisticRegression(solver = 'saga',multi_class='multinomial').fit(X_train_scale,y_train)

y_pred = clf.predict(X_train_scale)

accuracy = accuracy_score(y_test,y_pred)
print('Accuracy is',accuracy)


cap = cv2.VideoCapture(0)

while True:
    try:
        ret,frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        height,width = gray.shape
        upper_left = (int(width / 2 - 56), int(height / 2 - 56))
        bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        im_pil = Image.fromarray(roi)

        image_bw = im_pil.convert('L')
        image_bw_resized = image_bw.resize((28,28),Image.ANTIALIAS)

        image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
        pixel_filter = 20

        min_pixel = np.pixel(image_bw_resized_inverted, pixel_filter)

        image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)

        max_pixel = np.max(image_bw_resized_inverted)

        image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel

        test_sample = np.array(image_bw_resized_inverted_scaled,)
        test_pred = clf.predict(test_sample)
        print('Prediction:',test_pred)

        cv2.imshow('frame',gray)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    except Exception as e:
            pass

cap.release()
cv2.destroyAllWindows()



    








