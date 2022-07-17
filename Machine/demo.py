import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
import Convert

# Convert.convertedImagesTrain()
# Convert.convertedImagesTest()
# Convert.readFiles()


def flattenRGB(images):
    readImagesRGB = []

    for image in images:
        imgRGB = image

        contours, hierarchy = cv2.findContours(imgRGB, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image2 = cv2.drawContours(imgRGB.copy(), contours, -1, (0, 255, 0), 2)

        readImagesRGB.append(image2)

    lenImage = len(images)
    readImagesRGB = np.array(readImagesRGB)
    flattenedRGB = np.array(readImagesRGB).reshape(lenImage, -1)

    return flattenedRGB


def flattenGray(images):
    readImagesGray = []

    for image in images:
        imgG = image
        imgG = cv2.cvtColor(imgG, cv2.COLOR_BGR2GRAY)

        imgblurG = cv2.GaussianBlur(imgG, (0, 0), sigmaX=33, sigmaY=33)

        divide = cv2.divide(imgG, imgblurG, scale=255)

        thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        readImagesGray.append(morph)

    lenImage = len(images)
    readImagesGray = np.array(readImagesGray)
    flattenedGray = np.array(readImagesGray).reshape(lenImage, -1)

    return flattenedGray


def svmClass(features, output, featuresO, testO):
    clf = svm.SVC(kernel="poly")
    print("In SVM")
    clf.fit(features, output)
    print("training done")
    outA = clf.predict(featuresO)
    print("testing done")
    accS = accuracy_score(testO, outA) * 100
    preS = precision_score(testO, outA, average='micro')
    recS = recall_score(testO, outA, average='micro')
    print(testO[:5])
    print("____________________________________")
    print((outA[0:5]))
    out = accuracy_score(output, clf.predict(features)) * 100
    print("Train accuracy:", out)
    out2 = confusion_matrix(testO, outA)
    print(out2)
    return preS, recS, accS, outA


def logClass(features, output, featuresO, testO):
    classifier = LogisticRegression(max_iter=4500)
    print("In Logistic")
    classifier.fit(features, output)
    print("training done")
    outA = classifier.predict(featuresO)
    print("testing done")
    accL = accuracy_score(testO, outA) * 100
    preL = precision_score(testO, outA, average='micro')
    recL = recall_score(testO, outA, average='micro')
    print(testO[:5])
    print("____________________________________")
    print((outA[0:5]))
    out = accuracy_score(output, classifier.predict(features)) * 100
    print("Train accuracy:", out)
    out2 = confusion_matrix(testO, outA)
    print(out2)
    return preL, recL, accL, outA


def KnnClass(features, output, featuresO, testO):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(features, output)
    y_pred = classifier.predict(featuresO)
    accL = accuracy_score(testO, y_pred) * 100
    preL = precision_score(testO, y_pred, average='micro')
    recL = recall_score(testO, y_pred, average='micro')
    print(testO[:5])
    print("____________________________________")
    print((y_pred[0:5]))
    out = accuracy_score(output, classifier.predict(features)) * 100
    print("Train accuracy:", out)
    out2 = confusion_matrix(testO, y_pred)
    print(out2)
    return preL, recL, accL, y_pred


def naiveBayes(features, output, featuresO, testO):
    cnb = CategoricalNB()
    cnb.fit(features, output)
    y_pred = cnb.predict(featuresO)
    acc = accuracy_score(testO, y_pred) * 100
    pre = precision_score(testO, y_pred, average='micro')
    rec = recall_score(testO, y_pred, average='micro')
    print(testO[:5])
    print("____________________________________")
    print((y_pred[0:5]))
    out = accuracy_score(output, cnb.predict(features)) * 100
    print("Train accuracy:", out)
    out2 = confusion_matrix(testO, y_pred)
    print(out2)
    return pre, rec, acc, y_pred


def dtClass(features, output, featuresO, testO):
    dt = tree.DecisionTreeClassifier()
    dt = dt.fit(features, output)
    outA = dt.predict(featuresO)
    acc = accuracy_score(testO, outA) * 100
    pre = precision_score(testO, outA, average='micro')
    rec = recall_score(testO, outA, average='micro')
    print(testO[:5])
    print("____________________________________")
    print((outA[0:5]))
    out = accuracy_score(output, dt.predict(features)) * 100
    print("Train accuracy:", out)
    out2 = confusion_matrix(testO, outA)
    print(out2)
    return pre, rec, acc, outA


def RandomForest(features, output, featuresO, testO):
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(features, output)
    y_pred = clf.predict(featuresO)
    accL = accuracy_score(testO, y_pred) * 100
    preL = precision_score(testO, y_pred, average='micro')
    recL = recall_score(testO, y_pred, average='micro')
    print(testO[:5])
    print("____________________________________")
    print((y_pred[0:5]))
    out = accuracy_score(output, clf.predict(features)) * 100
    print("Train accuracy:", out)
    out2 = confusion_matrix(testO, y_pred)
    print(out2)
    return preL, recL, accL, y_pred


def modelsCall(trainDir, testDir):
    TrainData, TestData, labelOutputTrain, labelOutputTest = Convert.readFiles(trainDir, testDir)

    TrainRG, valRG, TrainlableOutRG, VallableoutRG = train_test_split(TrainData, labelOutputTrain,
                                                                      test_size=0.30, random_state=8)

    # TrainR = flattenRGB(TrainRG)
    # print("khlast flatten RGB train")
    TrainG = flattenGray(TrainRG)
    print("khlast flatten gray train")

    # ValR = flattenRGB(valRG)
    # print("khlast flatten RGB val")
    ValG = flattenGray(valRG)
    print("khlast flatten gray val")

    # preS, recS, accS, predOutS = svmClass(TrainG, TrainlableOutRG, ValG, VallableoutRG)
    # print("SVM Grey")
    # print(preS, recS, accS)  # , predOutS)
    # preSR, recSR, accSR, predOutSR = svmClass(TrainR, TrainlableOutRG, ValR, VallableoutRG)
    # print("SVM RGB")
    # print(preR, recR, accR)  #, predOutR)

    # preLog, recLog, accLog, predOutL = logClass(TrainG, TrainlableOutRG, ValG, VallableoutRG)
    # print("Logistic Regression Grey")
    # print(preLog, recLog, accLog)  # , predOutL)
    # preLogR, recLogR, accLogR, predOutLR = logClass(TrainR, TrainlableOutRG, ValR, VallableoutRG)
    # print("Logistic Regression RGB")
    # print(preLogR, recLogR, accLogR) #, predOutLR)

    # preK, recK, accK, predOutK = KnnClass(TrainG, TrainlableOutRG, ValG, VallableoutRG)
    # print("KNN Grey")
    # print(preK, recK, accK)  # , predOutK)
    # preKR, recKR, accKR, predOutKR = logClass(TrainR, TrainlableOutRG, ValR, VallableoutRG)
    # print("KNN RGB")
    # print(preKR, recKR, accKR) #, predOutKR)
    #
    # preN, recN, accN, predOutN = naiveBayes(TrainG, TrainlableOutRG, ValG, VallableoutRG)
    # print("Naive Grey")
    # print(preN, recN, accN)  # ,predOutN)
    # preNR, recNR, accNR, predOutNR = logClass(TrainR, TrainlableOutRG, ValR, VallableoutRG)
    # print("Naive RGB")
    # print(preNR, recNR, accNR) #, predOutNR)

    preD, recD, accD, predOutD = dtClass(TrainG, TrainlableOutRG, ValG, VallableoutRG)
    print("Decision Tree Grey")
    print(preD, recD, accD)  # ,predOutD)
    # preDR, recDR, accDR, predOutDR = logClass(TrainR, TrainlableOutRG, ValR, VallableoutRG)
    # print("Decision Tree RGB")
    # print(preDR, recDR, accDR) #, predOutDR)

    # preF, recF, accF, predOutF = RandomForest(TrainG, TrainlableOutRG, ValG, VallableoutRG)
    # print("Random Forest Gray")
    # print(preF, recF, accF) #,predOutF)
    # preFR, recFR, accFR, predOutFR = RandomForest(TrainG, labelOutputTrain, ValG, labelOutputTest)
    # print("Random Forest Grey")
    # print(preFR, recFR, accFR)  # ,predOutFR)


dirTrain = 'D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/PrePro_Data/Train/*'
dirTest = 'D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/PrePro_Data/Test/*'
modelsCall(dirTrain, dirTest)
