import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

# Define the image dimensions
img_height, img_width = 256, 256

folder_path = "E:/1-1/CIS500/Project 04/project-4-MuttakiIslamBismoyBracu18"


# Function to load and preprocess images from a folder
def load_and_preprocess_data(folder_path, label):
    data = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):  # Assuming your images are in jpg format
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_height, img_width))
            img = img / 255.0  # Normalize pixel values to [0, 1]

            data.append(img)
            labels.append(label)

    return np.array(data), np.array(labels)

# Define the paths to the folders containing cats and dogs images
cat = "E:/1-1/CIS500/Project 04/project-4-MuttakiIslamBismoyBracu18/Cats"
dog = "E:/1-1/CIS500/Project 04/project-4-MuttakiIslamBismoyBracu18/Dogs"

# Load and preprocess the data from the two folders
X_cat, y_cat = load_and_preprocess_data(cat, 0)  # 0 for Cats
X_dog, y_dog = load_and_preprocess_data(dog, 1)  # 1 for Dogs

# Concatenate the data and labels
X = np.concatenate((X_cat, X_dog), axis=0)
y = np.concatenate((y_cat, y_dog), axis=0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
hist= model.fit(X_train, y_train, epochs=34, validation_data=(X_test, y_test))

from matplotlib import pyplot as plt

fig = plt.figure()

plt.plot(hist.history['loss'], color='green', label='loss')
plt.plot(hist.history['val_loss'],color='red',label='val_loss')
fig.suptitle('Loss',fontsize=14)
plt.legend(loc="upper left")

plt.show()

fig = plt.figure()

plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'],color='orange',label='val_accuracy')
fig.suptitle('Accuracy',fontsize=14)
plt.legend(loc="upper left")

plt.show()

# Evaluate the model
predictions = model.predict(X_test)
y_pred = np.round(predictions)
y_true = y_test

# Print classification report and confusion matrix
print("Classification Report:\n", classification_report(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))


model.save(os.path.join('E:/1-1/CIS500/Project 04/Models','CatsAndDogs.keras'))

from PIL import Image

#import time

#time.sleep(30)


folder_path = 'E:/1-1/CIS500/Project 04/project-4-MuttakiIslamBismoyBracu18/Sample Images'
save_path = 'E:/1-1/CIS500/Project 04/project-4-MuttakiIslamBismoyBracu18/Output/Predicted_Cats'
save_path2 = 'E:/1-1/CIS500/Project 04/project-4-MuttakiIslamBismoyBracu18/Output/Predicted_Dogs'


# Define the image dimensions
img_height, img_width = 256, 256

for filename in os.listdir(folder_path):
    if filename.lower().endswith('.jpg'):  # Check if the file is a JPG
        img_path = os.path.join(folder_path, filename)
        image = Image.open(img_path)
        img = cv2.imread(img_path)

        if img is not None:
            # Resize and normalize the image for model prediction only
            img_for_prediction = cv2.resize(img, (img_height, img_width)) / 255.0
            img_for_prediction = np.expand_dims(img_for_prediction, axis=0)  # Add batch dimension

            # Perform inference with the model
            prediction = model.predict(img_for_prediction)
            if prediction > 0.5:
                    # Save the original image using the latitude and longitude in the filename
                    new_img_path = os.path.join(save_path2, filename)
                    cv2.imwrite(new_img_path, img)
                    print(f"{filename}: The probability of the picture has one or multiple dogs are {prediction*100}%")
            else:
                new_img_path = os.path.join(save_path, filename)
                cv2.imwrite(new_img_path, img)
                print(f"{filename}: The probability of the picture has one or multiple cats are {(1-prediction)*100}%")
        else:
            print(f"Error reading {filename}. The file may be corrupted or in an unsupported format.")