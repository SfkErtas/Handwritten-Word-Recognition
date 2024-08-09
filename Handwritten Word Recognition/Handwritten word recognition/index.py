import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

dataset_path = "C:/Users/safak/Downloads/dataset/data/words"

def load_and_preprocess_images(dataset_path):
    images = []
    labels = []

    for word_folder in os.listdir(dataset_path):
        
        word_folder_path = os.path.join(dataset_path, word_folder)

        if os.path.isdir(word_folder_path):
            print("Word folder:", word_folder)

            for file in os.listdir(word_folder_path):
            
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(word_folder_path, file)

                    print("Image path:", image_path)

                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                    if image is not None and not image.size == 0:
                
                        print(f"Original image shape: {image.shape}")

                        image = cv2.resize(image, (28, 28))
                        image = image / 255.0

                        label = word_folder

                        images.append(image)
                        labels.append(label)
                    else:
                        print(f"Skipping empty image: {image_path}")

    return np.array(images), np.array(labels)

images, labels = load_and_preprocess_images(dataset_path)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)


images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=None, stratify=labels)

label_encoder = LabelEncoder()

labels_train_encoded = label_encoder.fit_transform(labels_train)
labels_test_encoded = label_encoder.transform(labels_test)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(set(labels)), activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

images_train_reshaped = images_train.reshape(images_train.shape + (1,))

model.fit(images_train_reshaped, labels_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

images_test_reshaped = images_test.reshape(images_test.shape + (1,))

test_loss, test_accuracy = model.evaluate(images_test_reshaped, labels_test_encoded)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

sample_image = images_test_reshaped[0:1]
predictions = model.predict(sample_image)

decoded_predictions = label_encoder.inverse_transform(predictions.argmax(axis=1))

print("Actual Label:", labels_test[0])
print("Predicted Label:", decoded_predictions[0])
