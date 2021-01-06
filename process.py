import cv2, os
import numpy as np

from tensorflow.keras.utils import to_categorical

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

raw_path = 'dataset'
processed_path = 'processed'

categories = {
    0: 'without_mask',
    1: 'with_mask'
}

images = []
labels = []

for label, category in categories.items():

    folder_path = os.path.join(raw_path, category)
    image_names = os.listdir(folder_path)
        
    for image_name in image_names:
        image_path = os.path.join(folder_path, image_name)
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
            faces = face_cascade.detectMultiScale(image, 1.3, 5)
            if (len(faces) > 0):
                x,y,w,h = faces[0]
                image = image[y:y+w,x:x+w]
                image = cv2.resize(image, (100, 100))
                images.append(image)
                labels.append(label)
        except Exception as e:
            print('Error: ', e)

images = np.array(images)/255.0
images = np.reshape(images, (images.shape[0], 100, 100, 1))

labels = to_categorical(np.array(labels))

np.save('processed/images', images)
np.save('processed/labels', labels)

print("Raw data processed successfully.")