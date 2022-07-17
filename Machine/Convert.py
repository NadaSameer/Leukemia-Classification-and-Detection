import glob
from PIL import Image
import os
import random
import cv2


def readFiles(dirTrain, dirTest):
    directory = glob.glob(dirTrain)
    imageNamesTrain = []
    labelOutputTrain = []

    jpgTrain = []
    jpgTest = []

    for folder in directory:
        for file in glob.glob(folder + '/*.png') or glob.glob(folder + '/*.jpg'):
            print(file)
            jpgTrain.append(file)

    imageNamesMainTr = random.sample(jpgTrain, len(jpgTrain))

    for image in imageNamesMainTr:
        img = cv2.imread(image)
        img2 = cv2.resize(img, (124, 124))

        if img is not None:
            imageNamesTrain.append(img2)

        labels = image.split("/")
        name = labels[-1].split("\\")[1]

        labelOutputTrain.append(name)

    labelOutputTest = []
    imageNamesTest = []

    for folder in glob.glob(dirTest):
        for file in glob.glob(folder + '/*.png') or glob.glob(folder + '/*.jpg'):
            print(file)
            jpgTest.append(file)

    imageNamesMainTe = random.sample(jpgTest, len(jpgTest))

    for image in imageNamesMainTe:
        img = cv2.imread(image)
        img2 = cv2.resize(img, (124, 124))

        if img is not None:
            imageNamesTest.append(img2)

    for image in imageNamesMainTe:
        labels = image.split("/")
        name = labels[-1].split("\\")[1]
        labelOutputTest.append(name)

    for i in range(len(imageNamesMainTr)):
        print(imageNamesMainTr[i])
        print(labelOutputTrain[i])
        # print(imageNamesTrain[i])
    print("--------------------------------------------------------------------------------")
    for i in range(len(imageNamesMainTe)):
        print(imageNamesMainTe[i])
        print(labelOutputTest[i])
        # print(imageNamesTest[i])

    return imageNamesTrain, imageNamesTest, labelOutputTrain, labelOutputTest



def convertedImagesTrain():
    directory = glob.glob('D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/Train/*')
    jpgTrain = []
    bmpTrain = []
    tiffTrain = []

    for folder in directory:
        for file in glob.glob(folder + '/*.jpg'):
            jpgTrain.append(file)
        for file in glob.glob(folder + '/*.jpeg'):
            jpgTrain.append(file)
        for file in glob.glob(folder + '/*.bmp'):
            bmpTrain.append(file)
        for file in glob.glob(folder + '/*.tiff'):
            tiffTrain.append(file)
    #print(bmpTrain)
    for image in jpgTrain:
        img = Image.open(image)
        rgb_im = img.convert('RGB')
        name = image.split(".")[0]
        rgb_im.save(f"{name}.jpg")
        # print("IN JPG")

    for image in bmpTrain:
        img = Image.open(image)
        rgb_im = img.convert('RGB')
        name = image.split(".")[0]
        rgb_im.save(f"{name}.jpg")
        img.close()
        os.remove(image)
        # print("IN bmp")

    for image in tiffTrain:
        img = Image.open(image)
        rgb_im = img.convert('RGB')
        name = image.split(".")[0]
        rgb_im.save(f"{name}.jpg")
        img.close()
        os.remove(image)
        # print("in tiff")


def convertedImagesTest():
    imageNamesTest = []
    jpgTrain = []
    bmpTrain = []
    tiffTrain = []
    for folder in glob.glob('D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/Testing/*'):
        for file in glob.glob(folder + '/*.jpg'):
            imageNamesTest.append(file)
        for file in glob.glob(folder + '/*.jpeg'):
            jpgTrain.append(file)
        for file in glob.glob(folder + '/*.bmp'):
            bmpTrain.append(file)
        for file in glob.glob(folder + '/*.tiff'):
            tiffTrain.append(file)

    for image in imageNamesTest:
        img = Image.open(image)
        rgb_im = img.convert('RGB')
        name = image.split(".")[0]
        rgb_im.save(f"{name}.jpg")

    for image in bmpTrain:
        img = Image.open(image)
        rgb_im = img.convert('RGB')
        name = image.split(".")[0]
        rgb_im.save(f"{name}.jpg")
        img.close()
        os.remove(image)
        # print("IN bmp")

    for image in tiffTrain:
        img = Image.open(image)
        rgb_im = img.convert('RGB')
        name = image.split(".")[0]
        rgb_im.save(f"{name}.jpg")
        img.close()
        os.remove(image)
        # print("in tiff")
