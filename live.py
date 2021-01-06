import cv2, time
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = load_model('models/main.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

source = cv2.VideoCapture(0)

categories = {
    0: 'No Mask',
    1: 'Mask'
}

colors = {
    0: (0,0,255),
    1: (0,255,0)
}

last_label = 0
last_change = time.time()

while (True):
    ret, frame = source.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    for (x,y,w,h) in faces:
        face = image[y:y+w,x:x+w]
        normalized = cv2.resize(face, (100,100))/255.0
        reshaped = np.reshape(normalized, (1, 100, 100, 1))

        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]

        if (last_label != label):
            last_label = label
            last_change = time.time()

        elif (last_change + 3 < time.time()):
            last_change = time.time()
            last_label = label

            if (label == 0):
                #Deny entry to the person
                print("Deny entry")
            elif (label == 1):
                #Allow the person to enter
                print("Allow entry")

        cv2.rectangle(frame, (x,y), (x+w,y+h), colors[label], 2)
        cv2.rectangle(frame, (x,y-30), (x+w,y), colors[label], -2)
        cv2.putText(
          frame, categories[label] + " " + str(round(result[0][label]*100)) + "%", 
          (x + 10, y-10),
          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        
    cv2.imshow('Masketeer - @sheikhuzairhussain', frame)
    
    key=cv2.waitKey(1) 
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()