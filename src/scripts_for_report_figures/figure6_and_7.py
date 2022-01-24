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



convolutions_to_test = {
    "No_dropout":tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "1_dropout_0.5":tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "2_dropout_0.5":tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "3_dropout_0.5":tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ])
    }

'''
convolutions_to_test = {
    "1_dropout_0.5":tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "1_dropout_0.25":tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "1_dropout_0.75":tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.75),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    }
'''

results = {}

for model in convolutions_to_test.keys():
    results[model] = {"accuracy":None , "val_accuracy":None, "confusion_mat": None}


# Nous allons répéter ces étapes afin de faire des statistiques et trouver expérimentalement les meilleurs
# paramètres possibles

mean_acc = 0
mean_val_acc = 0
nb_iters = 20

for test_model in convolutions_to_test.keys():
    mean_acc=0
    mean_val_acc=0
    for iteration in range(nb_iters):
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
        BATCH_SIZE = 20
        SHUFFLE_BUFFER_SIZE = 100

        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
        test_dataset = test_dataset.batch(BATCH_SIZE)

        # DEFINITIONS DES DIFFERENTS MODELS QUE NOUS AVONS TESTES


        model=convolutions_to_test.get(test_model)

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        # Entrainement
        epochs=1
        history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)
        model.summary()
        mean_acc += history.history.get("accuracy")[-1]
        mean_val_acc += history.history.get("val_accuracy")[-1]

    mean_acc/=nb_iters
    mean_val_acc/=nb_iters
    predictions = np.argmax(model.predict(np.array(X_test)), axis=1)

    mat = tf.math.confusion_matrix(predictions, y_test)
    print("MEAN ACCURACY : ", mean_acc)
    print("MEAN VAL_ACCURACY : ", mean_val_acc)
    print("CONFUSION MATRIX : ", mat)

    results[test_model]["accuracy"] = mean_acc
    results[test_model]["val_accuracy"] = mean_val_acc
    results[test_model]["confusion_mat"] = mat

for model in results.keys():
    print("*********** {} ***********".format(model))
    print("     - Accuracy: ", results.get(model).get("accuracy"))
    print("     - Val_Accuracy: ", results.get(model).get("val_accuracy"))
    print("     - Conf Matrix: | {}, {} |\n                    | {}, {} |".format( results.get(model).get("confusion_mat")[0][0], results.get(model).get("confusion_mat")[0][1], results.get(model).get("confusion_mat")[1][0], results.get(model).get("confusion_mat")[1][1]))
    print(results.get(model).get("confusion_mat"))
    print()

fig=plt.figure()
# Accuracy et Val_Accuracy en fonction du nombre de couches de convolution
ax1=fig.add_subplot(111)


ax1.plot([0, 1, 2, 3],[results["No_dropout"]["accuracy"], results["1_dropout_0.5"]["accuracy"], results["2_dropout_0.5"]["accuracy"], results["3_dropout_0.5"]["accuracy"]])
ax1.plot([0, 1, 2, 3],[results["No_dropout"]["val_accuracy"], results["1_dropout_0.5"]["val_accuracy"], results["2_dropout_0.5"]["val_accuracy"], results["3_dropout_0.5"]["val_accuracy"]])
ax1.set_xlabel("Nombre de couches de dropout")
ax1.set_ylabel("Accuracy")
ax1.set_title('Accuracy en fonction du nombre de couches de dropout')
ax1.set_xticks([0, 1, 2, 3])


'''
ax1.plot([0.25, 0.5, 0.75],[results["1_dropout_0.25"]["accuracy"], results["1_dropout_0.5"]["accuracy"], results["1_dropout_0.75"]["accuracy"]])
ax1.plot([0.25, 0.5, 0.75],[results["1_dropout_0.25"]["val_accuracy"], results["1_dropout_0.5"]["val_accuracy"], results["1_dropout_0.75"]["val_accuracy"]])
ax1.set_xlabel("Ratio de dropout")
ax1.set_ylabel("Accuracy")
ax1.set_title('Accuracy en fonction du ratio de dropout')
ax1.set_xticks([0.25, 0.5, 0.75])
'''

plt.show()



