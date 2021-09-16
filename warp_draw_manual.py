import pickle
from correction_modules import get_points
# from math import factorial
import cv2
import numpy as np
import matplotlib.pyplot as plt


class MatchingError(Exception):
    pass


def resize_and_show(image, size=(800, 600)):
    image_resized = cv2.resize(image, size)
    cv2.imshow('image_resized', image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load(image_path, num=2,img=[]):
    for i in range(num):
        filename = image_path + f"{i}.png"
        image = cv2.imread(filename)
        img.append(image)
    return img


if __name__ == "__main__":
    image_path = "images/"

    img = load(image_path)
    img0 = img[0] #元画像
    img1 = img[1] #傾いた画像

    # 傾いた画像のhight/4,width/4
    h = int(img1.shape[0]/4)
    w = int(img1.shape[1]/4)

    # #元画像中の抽出領域の指定
    # with open('cut_points.pkl','rb') as f:
    #     x1,y1,x4,y4,shift=pickle.load(f)
    # print(x1,y1,x4,y4,shift)

    # manual入力
    shift=699/1290
    x1=284
    y1=304
    x4=3247
    y4=2278
    #(元画像のpx)/(傾いた画像のpx)
    # shift=0.542
    w0=int(w*shift)
    h0=int(h*shift)
    x2 = x1+w0
    x3 = x4-w0
    y2 = y1+h0
    y3 = y4-h0

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

    pts_num = 3
    # 特徴点の組み合わせを4つ入手
    pts = np.array([[[[0.] * 2] * pts_num]*2] * 4)
    for i in range(4):
        pts[i] = get_points(cut_img1[i], cut_img0[i])
        # pts[i] = get_points(cut_img0[i], cut_img1[i])

    src = pts[:, 0]
    dst = pts[:, 1]

    # 特徴点の位置を補正
    dst[1, :, 0] += w*3
    dst[2, :, 1] += h*3
    dst[3, :, 0] += w*3
    dst[3, :, 1] += h*3
    src[0, :, 0] += x1
    src[0, :, 1] += y1
    src[1, :, 0] += x3
    src[1, :, 1] += y1
    src[2, :, 0] += x1
    src[2, :, 1] += y3
    src[3, :, 0] += x3
    src[3, :, 1] += y3

    plt.figure(figsize=(12, 5*pts_num))
    for i in range(3):
        plt.subplot(pts_num, 2, 2*i+1)
        img1_plt = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        plt.imshow(img1_plt)
        x = dst[:, i, 0]
        y = dst[:, i, 1]
        # print(x.shape)
        plt.plot(x, y, marker="x", color="r", markersize=15, linestyle="None")
        # plt.savefig(image_path+"corner_pts0.png")
        # plt.show()

        plt.subplot(pts_num, 2, 2*i+2)
        img0_plt = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        plt.imshow(img0_plt)
        x = src[:, i, 0]
        y = src[:, i, 1]
        # print(x.shape)
        plt.plot(x, y, marker="x", color="r", markersize=15, linestyle="None")
    plt.savefig(image_path+"pts.png")
    # plt.show()
    # # getPerspectiveTransformの引数は
    # # float32,点の配列をさらに[]で囲ったものである必要がある
    src = np.array([src]).astype(np.float32)
    dst = np.array([dst]).astype(np.float32)

    # print(src)
    # print(dst)

    # list Mに1-3番目による変換行列を格納
    M = []
    for i in range(pts_num):
        M.append(cv2.getPerspectiveTransform(src[:, :, i], dst[:, :, i]))
        # print(M[i])

    M_diff = []
    M_diff.append(M[0]-M[1])
    M_diff.append(M[1]-M[2])
    M_diff.append(M[2]-M[0])
    # print(M_diff[0])
    # print(M_diff[1])
    # print(M_diff[2])
    print(np.max(np.abs(M_diff[0][:, :2])))
    print(np.max(np.abs(M_diff[1][:, :2])))
    print(np.max(np.abs(M_diff[2][:, :2])))

    M_marker = None
    M_diff_max = []
    for i in range(pts_num):
        M_diff_max.append(np.max(np.abs(M_diff[i][:, :2])))
        if M_diff_max[i] <= 0.1:
            M_marker = M[i]
            print(f'used M derives from M{i}')
            break

    if M_marker is None:
        raise MatchingError("適切な特徴点の組み合わせが見つかりませんでした")

    marker = cv2.imread(image_path+'marker.png', cv2.IMREAD_UNCHANGED)
    marker_warped = cv2.warpPerspective(
        marker, M_marker, (img1.shape[1], img1.shape[0]))
    resize_and_show(marker_warped, (800, 600))
    cv2.imwrite(image_path+'marker_warped.png', marker_warped)

    img_h, img_w = img1.shape[:2]

    alpha = marker_warped[:, :, 3]/255
    alpha = alpha.reshape(*alpha.shape, 1)
    # print(alpha.shape)
    drawn_img = img1
    drawn_img[:img_h, :img_w, :] = marker_warped[:img_h,
                                                 :img_w, :3]*alpha + img1[:img_h, :img_w, :]*(1-alpha)

    resize_and_show(drawn_img, (800, 600))
    cv2.imwrite(image_path+'1_drawn.png', drawn_img)
