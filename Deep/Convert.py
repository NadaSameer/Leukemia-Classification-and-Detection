import glob

from PIL import Image
import os
import random
import cv2
import numpy as np
import skimage.exposure


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
        img = RGBAtoRGB(image)
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

    # for i in range(len(imageNamesMainTr)):
    #     print(imageNamesMainTr[i])
    #     print(labelOutputTrain[i])
        # print(imageNamesTrain[i])
    print("--------------------------------------------------------------------------------")
    for i in range(len(imageNamesMainTe)):
        print(imageNamesMainTe[i])
        print(labelOutputTest[i])
        # print(imageNamesTest[i])

    return imageNamesMainTr, imageNamesMainTe, labelOutputTrain, labelOutputTest


def ReadImagesCNN(imageNames):
    readImagesGray = []
    readImagesRGB = []

    for image in imageNames:
        imgRGB = RGBAtoRGB(image)
        imgRGB = cv2.resize(imgRGB, (124, 124))
        readImagesRGB.append(imgRGB)

        imgG = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        imgG = cv2.resize(imgG, (124, 124))

        readImagesGray.append(imgG)
    readImagesRGB = np.array(readImagesRGB)
    readImagesGray = np.array(readImagesGray)
    print(readImagesRGB.shape)

    return readImagesRGB, readImagesGray


def convertedImagesTrain():
    directory = glob.glob('D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/NewNormalData/Train')
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
    # print(bmpTrain)
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
    for folder in glob.glob('D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/NewNormalData/Test'):
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
        img.close()
        os.remove(image)
        # print("in tiff")


def Background_Removal(image, path2, counter):
    # load image
    img = cv2.imread(image)
    # print(img.shape)

    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold
    thresh = cv2.threshold(gray, 11, 255, cv2.THRESH_BINARY)[1]

    # apply morphology to clean small spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    # get external contour
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)

    # draw white filled contour on black background as mas
    contour = np.zeros_like(gray)
    cv2.drawContours(contour, [big_contour], 0, 255, -1)

    # blur dilate image
    blur = cv2.GaussianBlur(contour, (5, 5), sigmaX=0, sigmaY=0, borderType=cv2.BORDER_DEFAULT)

    # stretch so that 255 -> 255 and 127.5 -> 0
    mask = skimage.exposure.rescale_intensity(blur, in_range=(127.5, 255), out_range=(0, 255))

    # put mask into alpha channel of input
    result = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask

    # save output
    cv2.imwrite(path2 + f"Z{counter}.png", result)


def RGBAtoRGB(img):
    imgg = cv2.imread(img)
    imgg = cv2.cvtColor(imgg, cv2.COLOR_BGRA2RGB)
    return imgg


def Change_image(directory_read, directory_save):
    count = 0
    for imagee in directory_read:
        count += 1
        Background_Removal(imagee, directory_save, count)
    print(count)

# directoryALL_read = glob.glob("D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Data/Train/ALL/*")
# directoryALL_save = "D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/PrePro_Data/Train/ALL/"

# directoryAML_read = glob.glob("D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Data/Train/AML/*")
# directoryAML_save = "D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/PrePro_Data/Train/AML/"

# directoryCML_read = glob.glob("D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Data/Train/CML/*")
# directoryCML_save = "D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/PrePro_Data/Train/CML/"

# directoryCLL_read = glob.glob("D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Data/Train/CLL/*")
# directoryCLL_save = "D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/PrePro_Data/Train/CLL/"

# Change_image(directoryALL_read, directoryALL_save)
# Change_image(directoryAML_read, directoryAML_save)
# Change_image(directoryCML_read, directoryCML_save)
# Change_image(directoryCLL_read, directoryCLL_save)

# dirTrain = 'D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Data/Train/*'
# dirTest = 'D:/Iseul/Education/College/(4)_2nd_Semester/Graduation Project/Data/Test/*'
# readFiles(dirTrain, dirTest)


# dirTrain = 'D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelWithAug/Train/*'
# dirTest = 'D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelWithAug/Test/*'
# readFiles(dirTrain, dirTest)