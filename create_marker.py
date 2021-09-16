import matplotlib.pyplot as plt
import cv2
import numpy as np
import os


def resize_and_show(image, size=(800, 600)):
    image_resized = cv2.resize(image, size)
    cv2.imshow('image_resized', image_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(image, save_dir, filename):
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, image)


image_dir = 'images_2'
image_file = '0.png'
image_path = os.path.join(image_dir, image_file)

image = cv2.imread(image_path)
print(image.shape)
# (1488,1964,3)

marker_size = (image.shape[0], image.shape[1], 4)
print(marker_size)
marker = np.zeros(marker_size)

start1 = (300, 2200)
end1 = (3500, 2200)
start2 = (2100, 400)
end2 = (2100, 4500)
thickness = 14
color = (0, 0, 255, 255)  # 四つ目はアルファチャンネル

# マーカー透過画像作成
cv2.line(
    marker, start1, end1,
    color, thickness=thickness,
    lineType=cv2.LINE_8
)

cv2.line(
    marker, start2, end2,
    color, thickness=thickness,
    lineType=cv2.LINE_8
)

# 元画像にマーカーを引く
cv2.line(
    image, start1, end1,
    color, thickness=thickness,
    lineType=cv2.LINE_8
)

cv2.line(
    image, start2, end2,
    color, thickness=thickness,
    lineType=cv2.LINE_8
)

resize_and_show(marker)
resize_and_show(image)

save_image(marker, image_path, 'marker.png')
save_image(image, image_path, '0_drawn.png')
