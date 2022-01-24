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


class_names = ["no_mask", "mask"]

img_height = 128
img_width = 128
class0_path = "..\\..\\img\\bounding_boxes_photos\\no_mask"
class1_path = "..\\..\\img\\bounding_boxes_photos\\mask"

# On charge nos images
class0_imgs = loadImages(class0_path)
class1_imgs = loadImages(class1_path)


# On définis les modèles à tester
convolutions_to_test = {
    "1_conv":tf.keras.Sequential([
        layers.Conv2D(2, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "2_conv":tf.keras.Sequential([
        layers.Conv2D(2, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "3_conv":tf.keras.Sequential([
        layers.Conv2D(2, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "4_conv":tf.keras.Sequential([
        layers.Conv2D(2, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "5_conv":tf.keras.Sequential([
        layers.Conv2D(2, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "1_conv_2*filters":tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "2_conv_2*filters":tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "3_conv_2*filters":tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "4_conv_2*filters":tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "5_conv_2*filters":tf.keras.Sequential([
        layers.Conv2D(4, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "1_conv_2*2*filters":tf.keras.Sequential([
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "2_conv_2*2*filters":tf.keras.Sequential([
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "3_conv_2*2*filters":tf.keras.Sequential([
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "4_conv_2*2*filters":tf.keras.Sequential([
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ]),
    "5_conv_2*2*filters":tf.keras.Sequential([
        layers.Conv2D(8, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(16, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, strides=(1, 1), activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(len(class_names), activation='softmax')
    ])
    }

results = {
    "1_conv": {"accuracy":None , "val_accuracy":None, "confusion_mat": None},
    "2_conv":{"accuracy":None , "val_accuracy":None, "confusion_mat": None},
    "3_conv":{"accuracy":None , "val_accuracy":None, "confusion_mat": None},
    "4_conv":{"accuracy":None , "val_accuracy":None, "confusion_mat": None},
    "5_conv":{"accuracy":None , "val_accuracy":None, "confusion_mat": None},
    "1_conv_2*filters":{"accuracy":None , "val_accuracy":None, "confusion_mat": None},
    "2_conv_2*filters":{"accuracy":None , "val_accuracy":None, "confusion_mat": None},
    "3_conv_2*filters":{"accuracy":None , "val_accuracy":None, "confusion_mat": None},
    "4_conv_2*filters":{"accuracy":None , "val_accuracy":None, "confusion_mat": None},
    "5_conv_2*filters":{"accuracy":None , "val_accuracy":None, "confusion_mat": None},
    "1_conv_2*2*filters":{"accuracy":None , "val_accuracy":None, "confusion_mat": None},
    "2_conv_2*2*filters":{"accuracy":None , "val_accuracy":None, "confusion_mat": None},
    "3_conv_2*2*filters":{"accuracy":None , "val_accuracy":None, "confusion_mat": None},
    "4_conv_2*2*filters":{"accuracy":None , "val_accuracy":None, "confusion_mat": None},
    "5_conv_2*2*filters":{"accuracy":None , "val_accuracy":None, "confusion_mat": None}
    }


# Nous allons répéter ces étapes afin de faire des statistiques et trouver expérimentalement les meilleurs
# paramètres possibles

mean_acc = 0
mean_val_acc = 0
nb_iters = 20

for test_model in convolutions_to_test.keys():
    mean_acc=0
    mean_val_acc=0
    
    for iteration in range(nb_iters):

        # Melange du dataset
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

        # Ici on choisis le modèle à tester
        model=convolutions_to_test.get(test_model)

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])

        # Entrainement
        epochs=10
        history = model.fit(train_dataset, validation_data=test_dataset, epochs=epochs)
        model.summary()
        mean_acc += history.history.get("accuracy")[-1]
        mean_val_acc += history.history.get("val_accuracy")[-1]

    mean_acc/=nb_iters
    mean_val_acc/=nb_iters
    predictions = np.argmax(model.predict(np.array(X_test)), axis=1)
    model.summary()

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

ax2=fig.add_subplot(511)
ax2.plot([1, 2, 4], [results["1_conv"]["accuracy"], results["1_conv_2*filters"]["accuracy"], results["1_conv_2*2*filters"]["accuracy"]])
ax2.plot( [1, 2, 4], [results["1_conv"]["val_accuracy"], results["1_conv_2*filters"]["val_accuracy"], results["1_conv_2*2*filters"]["val_accuracy"]],'r')
ax2.set_xlabel("Facteur multiplicateur du nombre de filtre")
ax2.set_ylabel("Accuracy")
ax2.set_title('Accuracy en fonction du facteur multiplicateur du nombre de filtre pour une seule couche de convolution')
ax2.set_xticks([1, 2, 4])

ax3=fig.add_subplot(512)
ax3.plot([1, 2, 4], [results["2_conv"]["accuracy"], results["2_conv_2*filters"]["accuracy"], results["2_conv_2*2*filters"]["accuracy"]])
ax3.plot([1, 2, 4], [results["2_conv"]["val_accuracy"], results["2_conv_2*filters"]["val_accuracy"], results["2_conv_2*2*filters"]["val_accuracy"]],'r')
ax3.set_xlabel("Facteur multiplicateur du nombre de filtre")
ax3.set_ylabel("Accuracy")
ax3.set_title('Accuracy en fonction du facteur multiplicateur du nombre de filtre pour 2 couches de convolution')
ax3.set_xticks([1, 2, 4])

ax4=fig.add_subplot(513)
ax4.plot([1, 2, 4], [results["3_conv"]["accuracy"], results["3_conv_2*filters"]["accuracy"], results["3_conv_2*2*filters"]["accuracy"]])
ax4.plot([1, 2, 4], [results["3_conv"]["val_accuracy"], results["3_conv_2*filters"]["val_accuracy"], results["3_conv_2*2*filters"]["val_accuracy"]],'r')
ax4.set_xlabel("Facteur multiplicateur du nombre de filtre")
ax4.set_ylabel("Accuracy")
ax4.set_title('Accuracy en fonction du facteur multiplicateur du nombre de filtre pour 3 couches de convolution')
ax4.set_xticks([1, 2, 4])

ax5=fig.add_subplot(514)
ax5.plot([1, 2, 4], [results["4_conv"]["accuracy"], results["4_conv_2*filters"]["accuracy"], results["4_conv_2*2*filters"]["accuracy"]])
ax5.plot([1, 2, 4], [results["4_conv"]["val_accuracy"], results["4_conv_2*filters"]["val_accuracy"], results["4_conv_2*2*filters"]["val_accuracy"]],'r')
ax5.set_xlabel("Facteur multiplicateur du nombre de filtre")
ax5.set_ylabel("Accuracy")
ax5.set_title('Accuracy en fonction du facteur multiplicateur du nombre de filtre pour 4 couches de convolution')
ax5.set_xticks([1, 2, 4])

ax6=fig.add_subplot(515)
ax6.plot([1, 2, 4],[results["5_conv"]["accuracy"], results["5_conv_2*filters"]["accuracy"], results["5_conv_2*2*filters"]["accuracy"]])
ax6.plot([1, 2, 4], [results["5_conv"]["val_accuracy"], results["5_conv_2*filters"]["val_accuracy"], results["5_conv_2*2*filters"]["val_accuracy"]],'r')
ax6.set_xlabel("Facteur multiplicateur du nombre de filtre")
ax6.set_ylabel("Accuracy")
ax6.set_title('Accuracy en fonction du facteur multiplicateur du nombre de filtre pour 5 couches de convolution')
ax6.set_xticks([1, 2, 4])

plt.show()


