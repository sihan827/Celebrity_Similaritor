import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    """
    modified version of cv2.imread for Korean directory path
    :param filename: image filename that you want to read
    :param flags: flags for cv2.imread, default cv2.IMREAD_COLOR
    :param dtype: dtype for cv2.imread, default np.uint8
    :return: image
    """
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename, img, params=None):
    """
    modified version of cv2.imwrite for Korean directory path
    :param filename: filename that you want to save
    :param img: image you want to save
    :param params: params for cv2.imwrite, default None
    """
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='wb') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def get_face(image):
    """
    crop face images from input image and return them using Haar cascade parameters
    :param image: input image you want to crop
    :return: list of cropped images
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.imshow(gray, cmap='gray')
    xml = './haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(xml)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print('Detected Face :', str(len(faces)))

    crops = []
    if len(faces):
        for (x, y, w, h) in faces:
            crops.append(image[y:y + h, x:x + w])

    return crops


def crop_celebrities_face_and_save(dir_path):
    """
    this function is for data 'celebrities_face'
    crop faces from crawled data and save them to 'celebrities_face' directory
    :param dir_path: directory path that contains none cropped images
    """
    n = dir_path.split('/')
    keyword = n[-1]

    dir = './celebrities_face/' + keyword
    if not os.path.exists(dir):
        os.makedirs(dir)

    file_list = os.listdir(dir_path)
    for file in file_list:
        file_path = dir_path + '/' + file
        img = imread(file_path)
        detected_faces = get_face(img)

        i = 0
        for face in detected_faces:
            imwrite(dir + '/' + file[:-4] + '_' + str(i) + '.jpg', face)
            i += 1


def crop_image_and_save(img_path):
    """
    crop faces from image in img_path and save them in same path
    :param img_path: image file path you want to crop
    """
    if not os.path.exists(img_path):
        print('No such image path exists!')
        return None
    file_name = img_path.split('/')[-1]
    index = img_path.find(file_name)
    cropped_path = img_path[:index]
    img = imread(img_path)
    detected_face = get_face(img)

    i = 0
    for face in detected_face:
        imwrite(cropped_path + file_name[:-4] + '_' + str(i) + '.jpg', face)
        i += 1


if __name__ == '__main__':
    # Celebrities Image Crop
    path = './celebrities'
    dir_list = os.listdir(path)
    for i in dir_list:
        p = path + '/' + i
        crop_celebrities_face_and_save(p)

