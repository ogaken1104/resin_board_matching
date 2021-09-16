from correction_modules import get_points
# from math import factorial
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle


class MatchingError(Exception):
    pass


def resize_and_show(image, size=(700, 500)):
    image_resized = cv2.resize(image, size)
    cv2.imshow('image_resized', image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load(image_path, img=[]):
    for i in range(2):
        filename = image_path + f"{i}.png"
        image = cv2.imread(filename)
        img.append(image)
        # print(img[i].shape)
    return img


if __name__ == "__main__":
    image_path = "images_2/"

    img = load(image_path)
    img0 = img[0] #元画像
    img1 = img[1] #傾いた画像

    # 傾いた画像のhight/4,width/4
    h = int(img1.shape[0]/4)
    w = int(img1.shape[1]/4)

    #元画像中の抽出領域の指定
    x1 = 830
    y1 = 1305
    x3 = 1900
    y3 = 3003

    #(元画像のpx)/(傾いた画像のpx)
    shift=0.410
    w0=int(w*shift)
    h0=int(h*shift)
    x2 = x1+w0
    x4 = x3+w0
    y2 = y1+h0
    y4 = y3+h0

    # 画像を4つに分割
    cut_img0 = [0]*4
    cut_img1 = [0]*4
    cut_img0[0] = img0[y1:y2, x1: x2]
    cut_img0[1] = img0[y1:y2, x3:x4]
    cut_img0[2] = img0[y3:y4, x1:x2]
    cut_img0[3] = img0[y3:y4, x3:x4]
    cut_img1[0] = img1[: h, : w]
    cut_img1[1] = img1[: h, w*3:]
    cut_img1[2] = img1[h*3:, : w]
    cut_img1[3] = img1[h*3:, w*3:]

    # for i,cut_img in enumerate(cut_img1):
    #     cv2.imwrite(image_path+f'cut_0{i}.png',cut_img) 
    for cut_img in cut_img0:
        resize_and_show(cut_img)

    with open('cut_points.pkl','wb') as f:
        pickle.dump((x1,y1,x3,y3,shift),f)
    