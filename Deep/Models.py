from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
import Convert
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import numpy
from sklearn.metrics import recall_score, precision_score
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2


def CNNModel(Train, labelTrain, Test, labelTest, imageDim):
    # batch_size = 64
    epochs = 3
    num_classes = 5
    num_filters = 32
    filter_size = 3
    pool_size = 2
    model = Sequential([
        Conv2D(num_filters, filter_size, input_shape=(124, 124, imageDim), padding="same"),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(num_filters, filter_size, padding="same"),
        MaxPooling2D(pool_size=pool_size),

        Flatten(),
        Dense(128, activation="relu"),  # Adding the Hidden layer
        Dropout(0.1, seed=2019),
        Dense(num_classes, activation='softmax'),  # ouput layer
    ])

    model.compile(
        'adam',
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    model.fit(
        Train,
        to_categorical(labelTrain),
        epochs=epochs,
        validation_data=(Test, to_categorical(labelTest)),

    )
    model.save("D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Saved Models/CNNModelAllMixedDataFully")

    model.save_weights(
        "D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Saved Models/CNNModelAllMixedDataFullyWeights")
    model.save("my_h5_model_CNN_AllMixedDataFully.h5")

    model.summary()

    classes = ['ALL', 'AML', 'CLL', 'CML', 'Normal']
    predictions = model.predict(Test)
    predictions = numpy.argmax(predictions, axis=1)
    print(sum([labelTest[i] == predictions[i] for i in range(len(predictions))]))
    print(len(labelTest))
    precision = precision_score(labelTest, predictions, average='micro')
    recall = recall_score(labelTest, predictions, average='micro')
    print(precision, recall)
    cm = confusion_matrix(labelTest, predictions)
    print(cm)


def CNNModel2(Train, labelTrain, Test, labelTest, imageDim):
    # batch_size = 64
    epochs = 3
    num_classes = 5
    num_filters = 64
    filter_size = 2
    pool_size = 2
    model = Sequential([
        Conv2D(num_filters, filter_size, input_shape=(124, 124, imageDim), padding="same"),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(num_filters, filter_size, padding="same"),
        MaxPooling2D(pool_size=pool_size),

        Flatten(),
        Dense(128, activation="relu"),  # Adding the Hidden layer
        Dropout(0.1, seed=2019),
        Dense(num_classes, activation='softmax'),  # ouput layer
    ])

    model.compile(
        'adam',
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    model.fit(
        Train,
        to_categorical(labelTrain),
        epochs=epochs,
        validation_data=(Test, to_categorical(labelTest)),

    )
    model.save("D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Saved Models/CNNModel2AllMixedDataFully")

    model.save_weights(
        "D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Saved Models/CNNModel2AllMixedDataFullyWeights")
    model.save("my_h5_model_CNN2_AllMixedDataFully.h5")

    model.summary()

    classes = ['ALL', 'AML', 'CLL', 'CML', 'Normal']
    predictions = model.predict(Test)
    predictions = numpy.argmax(predictions, axis=1)
    print(sum([labelTest[i] == predictions[i] for i in range(len(predictions))]))
    print(len(labelTest))
    precision = precision_score(labelTest, predictions, average='micro')
    recall = recall_score(labelTest, predictions, average='micro')
    print(precision, recall)
    cm = confusion_matrix(labelTest, predictions)
    print(cm)

def res50Built(Train, labelTrain, Test, labelTest, imageDim):
    epochs = 3
    model = Sequential([
        ResNet50(weights=None, include_top=True, input_shape=(124, 124, imageDim), classes=5)
    ])

    model.compile(
        'sgd',
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    model.fit(
        Train,
        to_categorical(labelTrain),
        epochs=epochs,
        validation_data=(Test, to_categorical(labelTest)),
    )
    model.save("D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/Saved Models/ModelResAllMixedDataFully")
    model.save_weights(
        "D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/Saved Models/ModelResAllMixedDataFullyWeights")
    model.save("my_h5_model_RES_AllMixedDataFully.h5")
    model.summary()

    classes = ['ALL', 'AML', 'CLL', 'CML', 'Normal']
    predictions = model.predict(Test)
    predictions = numpy.argmax(predictions, axis=1)
    print(sum([labelTest[i] == predictions[i] for i in range(len(predictions))]))
    print(len(labelTest))
    precision = precision_score(labelTest, predictions, average='micro')
    recall = recall_score(labelTest, predictions, average='micro')
    print(precision, recall)
    cm = confusion_matrix(labelTest, predictions)
    print(cm)


def AlexnetBuild(Train, labelTrain, Test, labelTest, imageDim):
    num_classes = 5
    model = keras.models.Sequential([
        keras.layers.Conv2D(filters=96, kernel_size=(5, 5), strides=(2, 2), activation='relu',
                            input_shape=(124, 124, imageDim)),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(4096, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
        optimizer=tf.optimizers.SGD(lr=0.001), )

    model.fit(Train,
              to_categorical(labelTrain),
              epochs=3,
              validation_data=(Test, to_categorical(labelTest)), )

    model.save("D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Saved Models/ModelAlexAllMixedDataFully")
    model.save_weights(
        "D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Saved Models/ModelAlexAllMixedDataFullyWeights")
    model.save("my_h5_model_ALEX_AllMixedDataFully.h5")

    model.summary()
    classes = ['ALL', 'AML', 'CLL', 'CML', 'Normal']
    predictions = model.predict(Test)
    predictions = numpy.argmax(predictions, axis=1)
    print(sum([labelTest[i] == predictions[i] for i in range(len(predictions))]))
    print(len(labelTest))
    precision = precision_score(labelTest, predictions, average='micro')
    recall = recall_score(labelTest, predictions, average='micro')
    print(precision, recall)
    cm = confusion_matrix(labelTest, predictions)
    print(cm)


def VGG16Built(Train, labelTrain, Test, labelTest, imageDim):
    epochs = 3
    num_classes = 5
    model = Sequential([
        VGG16(weights='imagenet', include_top=False, input_shape=(124, 124, imageDim)),
        Flatten(),
        Dense(1010, activation="relu"),  # Adding the Hidden layer
        Dropout(0.2, seed=2022),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        'sgd',
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    model.fit(
        Train,
        to_categorical(labelTrain),
        epochs=epochs,
        validation_data=(Test, to_categorical(labelTest)),
    )

    model.save("D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Saved Models/Model1VGG16NewData")
    model.save_weights(
        "D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Saved Models/Model1VGG16NewDataWeights")
    model.save("my_h5_model_VGG.h5")

    model.summary()
    classes = ['ALL', 'AML', 'CLL', 'CML', 'Normal']
    predictions = model.predict(Test)
    predictions = numpy.argmax(predictions, axis=1)
    print(sum([labelTest[i] == predictions[i] for i in range(len(predictions))]))
    print(len(labelTest))
    precision = precision_score(labelTest, predictions, average='micro')
    recall = recall_score(labelTest, predictions, average='micro')
    print(precision, recall)
    cm = confusion_matrix(labelTest, predictions)
    print(cm)


def MobilenetModel(Train, labelTrain, Test, labelTest, imageDim):
    epochs = 3
    classes = 5
    mobilenet = MobileNetV2(
        input_shape=(124, 124, imageDim),
        include_top=False,
        weights=None,
    )
    model = Sequential([
        mobilenet,
        GlobalAveragePooling2D(),
        Dropout(0.1),
        Dense(classes, activation='softmax')
    ])

    model.compile(
        'sgd',
        loss='categorical_crossentropy',
        metrics=[tf.keras.metrics.AUC(name="accuracy"), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )

    model.fit(
        Train,
        to_categorical(labelTrain),
        epochs=epochs,
        validation_data=(Test, to_categorical(labelTest)),

    )

    model.save("D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Saved Models/MobileNetNewData")

    model.save_weights(
        "D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Saved Models/MobileNetWeightsNewData")
    model.save("my_h5_model_MobileNetNewData.h5")

    model.summary()

    classes = ['ALL', 'AML', 'CLL', 'CML', 'Normal']
    predictions = model.predict(Test)
    predictions = numpy.argmax(predictions, axis=1)
    print(sum([labelTest[i] == predictions[i] for i in range(len(predictions))]))
    print(len(labelTest))
    precision = precision_score(labelTest, predictions, average='micro')
    recall = recall_score(labelTest, predictions, average='micro')
    print(precision, recall)
    cm = confusion_matrix(labelTest, predictions)
    print(cm)


def modelsCall(trainDir, testDir):
    TrainData, TestData, labelOutputTrain, labelOutputTest = Convert.readFiles(trainDir, testDir)

    TrainR, TrainG = Convert.ReadImagesCNN(TrainData)
    TestR, TestG = Convert.ReadImagesCNN(TestData)
    outProcessTrain = preprocessing.LabelEncoder()
    outProcessTest = preprocessing.LabelEncoder()

    outProcessTest.fit(labelOutputTest)
    LabelTest = outProcessTest.transform(labelOutputTest)

    outProcessTrain.fit(labelOutputTrain)
    LabelTrain = outProcessTrain.transform(labelOutputTrain)

    # CNNModel(TrainR, LabelTrain, TestR, LabelTest, int(3))
    # CNNModel2(TrainR, LabelTrain, TestR, LabelTest, int(3))
    AlexnetBuild(TrainR, LabelTrain, TestR, LabelTest, int(3))
    res50Built(TrainR, LabelTrain, TestR, LabelTest, int(3))
    # VGG16Built(TrainR, LabelTrain, TestR, LabelTest, int(3))
    # MobilenetModel(TrainR, LabelTrain, TestR, LabelTest, int(3))


dirTrain = 'D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/AllMixedData/Train/*'
dirTest = 'D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/AllMixedData/Test/*'
modelsCall(dirTrain, dirTest)
