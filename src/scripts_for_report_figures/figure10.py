import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.image as mpimg
from matplotlib import cm
import matplotlib.pyplot as plt
import random


def loadImages(path):
    # On charge les images, les met dans une liste
    image_files = sorted([os.path.join(path, file) for file in os.listdir(path)])
    return [np.array(cv2.imread(i, cv2.IMREAD_UNCHANGED))[:, :, :3]/255 for i in image_files]

# Affiche une image
def display_one(a, title1 = "Original"):
    plt.imshow(a), plt.title(title1)
    plt.xticks([]), plt.yticks([])
    plt.show()

class_names = ["no_mask", "mask"]

img_height = 128
img_width = 128
class0_path = "..\\..\\img\\bounding_boxes_photos\\no_mask"
class1_path = "..\\..\\img\\bounding_boxes_photos\\mask"

class0_imgs = loadImages(class0_path)
class1_imgs = loadImages(class1_path)



convolutions_to_test = {"1_dropout_0.5":tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ])
    }



results = []

# Nous allons répéter ces étapes afin de faire des statistiques et trouver expérimentalement les meilleurs
# paramètres possibles




iterations = 20
for e in range(1,11):
    accuracy = 0
    precision = 0
    recall = 0
    f1score = 0

    for iteration in range(iterations):

        random.shuffle(class0_imgs)
        random.shuffle(class1_imgs)
        num_class0 = len(class0_imgs)
        num_class1 = len(class1_imgs)
        print("Nombre d'images de classe 0: ", num_class0)
        print("Nombre d'images de classe 1: ", num_class1)


        # Ici on découpe notre dataset pour l'entrainement et la validation
        X_train0 = class0_imgs[:int(num_class0*0.70)]
        X_train1 = class1_imgs[:int(num_class1*0.70)]
        y_train = [0 for i in range(len(X_train0))] + [1 for i in range(len(X_train1))]
        X_train = X_train0 + X_train1

        X_test0 = class0_imgs[int(num_class0*0.70):]
        X_test1 = class1_imgs[int(num_class1*0.70):]
        y_test = [0 for i in range(len(X_test0))] + [1 for i in range(len(X_test1))]
        X_test = X_test0 + X_test1

        # On transforme nos deux datasets en tensors pour tensorflow
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


        # On choisis une taille de batch et on mélange nos datasets à l'aide de méthodes tensorflow
        BATCH_SIZE = 2
        SHUFFLE_BUFFER_SIZE = 100

        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        test_dataset = test_dataset.batch(BATCH_SIZE)

        
        model=tf.keras.Sequential([
                layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
                layers.MaxPooling2D(),
                layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
                layers.MaxPooling2D(),
                layers.Dropout(0.5),
                layers.Flatten(),
                layers.Dense(len(class_names), activation='softmax')
            ])

        model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

        # Entrainement
        epochs=5*e
        history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs, verbose=0)


        predictions = np.argmax(model.predict(np.array(X_test)), axis=1)
        mat = tf.math.confusion_matrix(predictions, y_test).numpy()
        print("CONFUSION MATRIX : ", mat)

        pres = mat[1][1]/(mat[1][1]+mat[0][1])
        rec = mat[1][1]/(mat[1][1]+mat[1][0])
        accuracy += (mat[0][0] + mat[1][1])/mat.sum()
        precision += pres
        recall += rec
        f1score += 2*( (pres*rec) / (pres+rec))
    results.append([accuracy/iterations, precision/iterations, recall/iterations, f1score/iterations])

    print("Accuracy: ", accuracy/iterations)
    print("Precision: ", precision/iterations)
    print("recall: ", recall/iterations)
    print("f1score: ", f1score/iterations)
    



print(results)
fig=plt.figure()
# Accuracy et Val_Accuracy en fonction du nombre de couches de convolution
ax1=fig.add_subplot(111)

for metric in range(4):
    if metric == 0:
        col = 'b'
    if metric == 1:
        col = 'g'
    if metric == 2:
        col = 'r'
    if metric == 3:
        col = 'y'
    ax1.plot([5, 10, 15, 20, 25, 30, 35, 40, 45, 50],[results[0][metric], results[1][metric], results[2][metric],
                                                      results[3][metric], results[4][metric], results[5][metric],
                                                      results[6][metric], results[7][metric], results[8][metric],
                                                      results[9][metric]], col)
ax1.set_xlabel("Nombre d'epoch d'entrainement")
ax1.set_ylabel("metriques")
ax1.set_title("Accuracy, precision, recall et f1-score en fonction du nombre d'epoch d'entrainement")
ax1.set_xticks([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

plt.show()


