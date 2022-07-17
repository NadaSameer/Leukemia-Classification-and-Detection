import cv2
import numpy as np
import glob


def Flip(img, counter, path):
    counter += 1
    image = cv2.imread(img)

    img_rotate_90_clockwise = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(path + f"Z{counter}.jpg", img_rotate_90_clockwise)
    counter += 1

    img_rotate_90_counterclockwise = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(path + f"Z{counter}.jpg", img_rotate_90_counterclockwise)
    counter += 1

    img_rotate_180 = cv2.rotate(image, cv2.ROTATE_180)
    cv2.imwrite(path + f"Z{counter}.jpg", img_rotate_180)
    return counter


def Rotate(img, counter, path):
    counter += 1
    image = cv2.imread(img)
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
    rotated45 = cv2.warpAffine(image, M, (w, h))
    cv2.imwrite(path + f"Z{counter}.jpg", rotated45)
    counter += 1

    M = cv2.getRotationMatrix2D((cX, cY), 30, 1.0)
    rotated30 = cv2.warpAffine(image, M, (w, h))
    cv2.imwrite(path + f"Z{counter}.jpg", rotated30)
    counter += 1

    M = cv2.getRotationMatrix2D((cX, cY), 60, 1.0)
    rotated60 = cv2.warpAffine(image, M, (w, h))
    cv2.imwrite(path + f"Z{counter}.jpg", rotated60)
    counter += 1

    M = cv2.getRotationMatrix2D((cX, cY), 115, 1.0)
    rotated115 = cv2.warpAffine(image, M, (w, h))
    cv2.imwrite(path + f"Z{counter}.jpg", rotated115)
    return counter


def Translate(img, counter, path):
    counter += 1
    image = cv2.imread(img)

    translation_matrix1 = np.float32([[1, 0, 120], [0, 1, 50]])
    shifted1 = cv2.warpAffine(image, translation_matrix1, (image.shape[1], image.shape[0]))
    cv2.imwrite(path + f"Z{counter}.jpg", shifted1)
    counter += 1

    translation_matrix2 = np.float32([[1, 0, 5], [0, 1, 100]])
    shifted2 = cv2.warpAffine(image, translation_matrix2, (image.shape[1], image.shape[0]))
    cv2.imwrite(path + f"Z{counter}.jpg", shifted2)
    counter += 1

    translation_matrix3 = np.float32([[1, 0, 40], [0, 1, 50]])
    shifted3 = cv2.warpAffine(image, translation_matrix3, (image.shape[1], image.shape[0]))
    cv2.imwrite(path + f"Z{counter}.jpg", shifted3)
    return counter


def Scale(img, counter, path):
    counter += 1
    image = cv2.imread(img)

    scale_percent = 60
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path + f"Z{counter}.jpg", resized)
    return counter


def Crop(img, counter, path):
    counter += 1
    image = cv2.imread(img)

    height, width = image.shape[0:2]
    startRow = int(height * .15)
    startCol = int(width * .15)
    endRow = int(height * .85)
    endCol = int(width * .85)

    croppedImage = image[startRow:endRow, startCol:endCol]
    cv2.imwrite(path + f"Z{counter}.jpg", croppedImage)
    return counter


def Noise(img, counter, path):
    counter += 1
    image = cv2.imread(img)
    mean = 0
    var = 10
    sigma = var ** 1.2
    shapee = image.shape
    print(shapee)
    gaussian = np.random.normal(mean, sigma, shapee[0:2])
    noisy_image = np.zeros(image.shape, np.float32)

    if len(image.shape) == 2:
        noisy_image = image + gaussian
    else:
        noisy_image[:, :, 0] = image[:, :, 0] + gaussian
        noisy_image[:, :, 1] = image[:, :, 1] + gaussian
        noisy_image[:, :, 2] = image[:, :, 2] + gaussian

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    cv2.imwrite(path + f"Z{counter}.jpg", noisy_image)
    return counter


def Saturation(img, counter, path):
    counter += 1
    image = cv2.imread(img)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = image[:, :, 2]
    v = np.where(v <= 255 - 150, v + 150, 255)
    image[:, :, 2] = v

    saturated = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    cv2.imwrite(path + f"Z{counter}.jpg", saturated)
    return counter


def Brightness(img, counter, path):
    counter += 1
    image = cv2.imread(img)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v += 25

    final_hsv = cv2.merge((h, s, v))
    bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(path + f"Z{counter}.jpg", bright)
    counter += 1

    final_hsv = cv2.merge((h, s, v + 10))
    bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(path + f"Z{counter}.jpg", bright)
    return counter


def ColorChange(img, counter, path):
    counter += 1
    image = cv2.imread(img)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sensitivity = 30
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    image[mask > 0] = (170, 170, 170)

    cv2.imwrite(path + f"Z{counter}.jpg", image)
    return counter


def Contrast(img, counter, path):
    counter += 1
    image = cv2.imread(img)

    contrast_img = cv2.convertScaleAbs(image, alpha=1.3, beta=0)
    cv2.imwrite(path + f"Z{counter}.jpg", contrast_img)
    counter += 1

    contrast_img = cv2.convertScaleAbs(image, alpha=1.8, beta=0.2)
    cv2.imwrite(path + f"Z{counter}.jpg", contrast_img)
    return counter

def augmentation():

    # directoryCML = glob.glob("D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelCLL_CML_Normal/Train/AML/*")
    # pathCML = "D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelCLL_CML_Normal/Train/AML/"
    #
    # directoryCLL = glob.glob("D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelCLL_CML_Normal/Train/ALL/*")
    # pathCLL = "D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelCLL_CML_Normal/Train/ALL/"
    #
    # directoryNormal = glob.glob(
    #     "D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelCLL_CML_Normal/Train/Normal/*")
    # pathNormal = "D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelCLL_CML_Normal/Train/Normal/"
    #
    # counterCML = 0
    # for image in directoryCML:
    #     ret1 = Flip(image, counterCML, pathCML)
    #     ret2 = Rotate(image, ret1, pathCML)
    #     ret3 = Translate(image, ret2, pathCML)
    #     ret4 = Scale(image, ret3, pathCML)
    #     ret5 = Crop(image, ret4, pathCML)
    #     ret6 = Noise(image, ret5, pathCML)
    #     ret7 = Saturation(image, ret6, pathCML)
    #     ret8 = Brightness(image, ret7, pathCML)
    #     ret9 = ColorChange(image, ret8, pathCML)
    #     ret10 = Contrast(image, ret9, pathCML)
    #     counterCML = ret10
    #
    # counterCLL = 0
    # for image in directoryCLL:
    #     ret1 = Flip(image, counterCLL, pathCLL)
    #     ret2 = Rotate(image, ret1, pathCLL)
    #     ret3 = Translate(image, ret2, pathCLL)
    #     ret4 = Scale(image, ret3, pathCLL)
    #     ret5 = Crop(image, ret4, pathCLL)
    #     ret6 = Noise(image, ret5, pathCLL)
    #     ret7 = Saturation(image, ret6, pathCLL)
    #     ret8 = Brightness(image, ret7, pathCLL)
    #     ret9 = ColorChange(image, ret8, pathCLL)
    #     ret10 = Contrast(image, ret9, pathCLL)
    #     counterCLL = ret10
    #
    # for image in directoryNormal:
    #     ret1 = Flip(image, counterCLL, pathNormal)
    #     ret2 = Rotate(image, ret1, pathNormal)
    #     ret3 = Translate(image, ret2, pathNormal)
    #     ret4 = Scale(image, ret3, pathNormal)
    #     ret5 = Crop(image, ret4, pathNormal)
    #     ret6 = Noise(image, ret5, pathNormal)
    #     ret7 = Saturation(image, ret6, pathNormal)
    #     ret8 = Brightness(image, ret7, pathNormal)
    #     ret9 = ColorChange(image, ret8, pathNormal)
    #     ret10 = Contrast(image, ret9, pathNormal)
    #     counterCLL = ret10

    # TRAIN
    directoryCML = glob.glob("D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelCLL_CML_Normal/Train/AML/*")
    pathCML = "D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelCLL_CML_Normal/Train/AML/"

    directoryCLL2 = glob.glob("D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelCLL_CML_Normal/Train/ALL/*")
    pathCLL2 = "D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelCLL_CML_Normal/Train/ALL/"

    directoryNormal = glob.glob(
        "D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelCLL_CML_Normal/Train/Normal/*")
    pathNormal = "D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/ModelCLL_CML_Normal/Train/Normal/"

    counterCML = 0
    for image in directoryCML:
        ret1 = Flip(image, counterCML, pathCML)
        ret2 = Rotate(image, ret1, pathCML)
        ret3 = Translate(image, ret2, pathCML)
        ret4 = Scale(image, ret3, pathCML)
        ret5 = Crop(image, ret4, pathCML)
        ret6 = Noise(image, ret5, pathCML)
        ret7 = Saturation(image, ret6, pathCML)
        ret8 = Brightness(image, ret7, pathCML)
        ret9 = ColorChange(image, ret8, pathCML)
        ret10 = Contrast(image, ret9, pathCML)
        counterCML = ret10
        if counterCML > 550:
            break
    counterCLL = 0
    for image in directoryCLL2:
        ret1 = Flip(image, counterCLL, pathCLL2)
        ret2 = Rotate(image, ret1, pathCLL2)
        ret3 = Translate(image, ret2, pathCLL2)
        ret4 = Scale(image, ret3, pathCLL2)
        ret5 = Crop(image, ret4, pathCLL2)
        ret6 = Noise(image, ret5, pathCLL2)
        ret7 = Saturation(image, ret6, pathCLL2)
        ret8 = Brightness(image, ret7, pathCLL2)
        ret9 = ColorChange(image, ret8, pathCLL2)
        ret10 = Contrast(image, ret9, pathCLL2)
        counterCLL = ret10
        if counterCLL > 550:
            break
    counterNormal = 0
    for image in directoryNormal:
        ret1 = Flip(image, counterNormal, pathNormal)
        ret2 = Rotate(image, ret1, pathNormal)
        ret3 = Translate(image, ret2, pathNormal)
        ret4 = Scale(image, ret3, pathNormal)
        ret5 = Crop(image, ret4, pathNormal)
        ret6 = Noise(image, ret5, pathNormal)
        ret7 = Saturation(image, ret6, pathNormal)
        ret8 = Brightness(image, ret7, pathNormal)
        ret9 = ColorChange(image, ret8, pathNormal)
        ret10 = Contrast(image, ret9, pathNormal)
        counterNormal = ret10
        if counterNormal > 550:
            break

    # TEST

    # directoryCML2 = glob.glob("D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/Testing/CML/*")
    # pathCML2 = "D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/Testing/CML/"
    #
    # directoryCLL2 = glob.glob("D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/Testing/CLL/*")
    # pathCLL2 = "D:/Iseul/Education/College/(4)_1st_Semester/Graduation Project/Testing/CLL/"
    #
    # for image in directoryCML2[6:]:
    #     ret1 = Flip(image, counterCML, pathCML2)
    #     ret2 = Rotate(image, ret1, pathCML2)
    #     ret3 = Translate(image, ret2, pathCML2)
    #     ret4 = Scale(image, ret3, pathCML2)
    #     ret5 = Crop(image, ret4, pathCML2)
    #     ret6 = Noise(image, ret5, pathCML2)
    #     ret7 = Saturation(image, ret6, pathCML2)
    #     ret8 = Brightness(image, ret7, pathCML2)
    #     ret9 = ColorChange(image, ret8, pathCML2)
    #     ret10 = Contrast(image, ret9, pathCML2)
    #     counterCML = ret10
    #     if counterCML > 210:
    #         break
    #
    # for image in directoryCLL2[3:]:
    #     ret1 = Flip(image, counterCLL, pathCLL2)
    #     ret2 = Rotate(image, ret1, pathCLL2)
    #     ret3 = Translate(image, ret2, pathCLL2)
    #     ret4 = Scale(image, ret3, pathCLL2)
    #     ret5 = Crop(image, ret4, pathCLL2)
    #     ret6 = Noise(image, ret5, pathCLL2)
    #     ret7 = Saturation(image, ret6, pathCLL2)
    #     ret8 = Brightness(image, ret7, pathCLL2)
    #     ret9 = ColorChange(image, ret8, pathCLL2)
    #     ret10 = Contrast(image, ret9, pathCLL2)
    #     counterCLL = ret10
    #     if counterCLL > 105:
    #         break

# augmentation()