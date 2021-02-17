import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename, img, params=None):
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


def crop_images_and_save(dir_path):
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

        n = 0
        for face in detected_faces:
            imwrite(dir + '/' + file[:-4] + '_' + str(n) + '.jpg', face)


if __name__ == '__main__':
    # img = imread('./female/남보라/남보라_1.png')
    # img = imread('./song.jpg')
    #
    # detected_faces = get_face(img)
    #
    # plt.imshow(cv2.cvtColor(detected_faces[0], cv2.COLOR_BGR2RGB), cmap='gray')
    # plt.show()
    #
    # cv2.imwrite('./song2.jpg', detected_faces[0])
    
    # crop_images_and_save('./male/정해인')

    # Female Image Crop
    path = 'female'
    dir_list = os.listdir(path)
    for i in dir_list:
        p = path + '/' + i
        crop_images_and_save(p)

    # Male Image Crop
    path = 'male'
    dir_list = os.listdir(path)
    for i in dir_list:
        p = path + '/' + i
        crop_images_and_save(p)




