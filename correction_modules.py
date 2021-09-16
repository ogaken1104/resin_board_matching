import cv2 as cv
import numpy as np


# ndarray形式の画像から特徴点の組み合わせを得る関数
def get_points(img1, img2):

    # 配列をグレースケールに変更
    img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    # 特徴量の検出
    akaze = cv.AKAZE_create()
    # 特徴点 Key Points kp1, kp2 (座標)
    # 特徴量記述子 Feature Description des1, des2
    kp1, des1 = akaze.detectAndCompute(img1_gray, None)
    kp2, des2 = akaze.detectAndCompute(img2_gray, None)

    # img1_key = cv.drawKeypoints(img1_gray, kp1, None)
    # img2_key = cv.drawKeypoints(img2_gray, kp2, None)

    # # 特徴点の描画
    # img1_key = cv.resize(img1_key, (680, 480))
    # cv.imshow("img1_key", img1_key)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    # img2_key = cv.resize(img2_key, (680, 480))
    # cv.imshow("img2_key", img2_key)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # 特徴量を総当たりでマッチングします。
    # Brute-Force Matcher作成
    match = cv.BFMatcher()
    # マッチング度合いが高い順に二つ (k=2) 取得します。
    matches = match.knnMatch(des2, des1, k=2)

    # 閾値がi*0.0001のときの特徴点の組み合わせをgoodに格納する関数
    def matching(i):
        good = []
        for m, n in matches:
            if m.distance < i*0.001 * n.distance:
                good.append(m)
        return good

    # マッチ度の高い特徴点を上から二つ取得
    i = 1
    MATCH_COUNT = 3

    good = matching(i)
    # goodの要素(条件を満たす点の組み合わせ)が3組未満の時閾値をどんどん大きくしていく
    while len(good) < MATCH_COUNT:
        i += 1
        print(len(good))
        good = matching(i)

    # 特徴点の組み合わせが3対よりも多かったときに3対にする
    good = good[:3]

    # img2の特徴点の座標
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good])
    # img1の特徴点の座標
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good])

    # マッチング結果を可視化して確認
    draw_params = dict(matchColor=None, singlePointColor=None, flags=2)

    # print(src_pts)
    # print(dst_pts)

    drawmatches = cv.drawMatches(
        img2, kp2, img1, kp1, good, None, **draw_params)
    drawmatches = cv.resize(drawmatches, (680, 480))
    cv.imshow("drawmatches", drawmatches)
    cv.waitKey(0)
    cv.destroyAllWindows()

    points = np.array([[[0.] * 2] * 3] * 2)

    points[0] = src_pts
    points[1] = dst_pts

    return points
# 2枚の入力画像をnumpy配列で
# 特徴点の座標をnumpy配列のリストで
# 受け取り、合成画像のnumpy配列を返す関数


def stitching_img(img1, img2, pts=[]):
    if len(pts) == 0:
        pts = get_points(img1, img2)

    # img1がimg2の左のとき(img2の特徴点のx座標<img1の特徴点のx座標)
    if np.all(pts[0][:, 0] < pts[1][:, 0]):
        i = 2
        af = cv.getAffineTransform(pts[0], pts[1])
        img_warped = cv.warpAffine(
            img2, af, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        img_stitched = img_warped.copy()
        img_stitched[: img1.shape[0], : img1.shape[1]] = img1
    # img1がimg2の右のとき(img2の特徴点のx座標>img1の特徴点のx座標)
    elif np.all(pts[0][:, 0] > pts[1][:, 0]):
        i = 1
        af = cv.getAffineTransform(pts[1], pts[0])
        img_warped = cv.warpAffine(
            img1, af, (img1.shape[1] + img2.shape[1], img2.shape[0]))
        img_stitched = img_warped.copy()
        img_stitched[:img2.shape[0], :img2.shape[1]] = img2
    else:
        print("特徴点の位置関係に異常あり")

    # どちらの画像を移動させたかを出力
    print(f"warped_image : {i}")

    # 余分な 0 領域を削除
    # 画像の端から順に、要素がすべて0の行、列を消す
    def trim(frame):
        if np.sum(frame[0]) == 0:
            return trim(frame[1:])
        if np.sum(frame[-1]) == 0:
            return trim(frame[:-2])
        if np.sum(frame[:, 0]) == 0:
            return trim(frame[:, 1:])
        if np.sum(frame[:, -1]) == 0:
            return trim(frame[:, :-2])
        return frame

    img_stitched_trimmed = trim(img_stitched)

    # 各種画像の表示
    # Affine変換で移動された画像
    img_warped = cv.resize(img_warped, (680, 480))
    cv.imshow('img_warped', img_warped)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 合成された画像
    img_stitched = cv.resize(img_stitched, (680, 480))
    cv.imshow('img_stitched', img_stitched)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # トリミングされた合成画像
    img = cv.resize(img_stitched_trimmed, (680, 480))
    cv.imshow('img_stitched_trimmed', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return img_stitched_trimmed
