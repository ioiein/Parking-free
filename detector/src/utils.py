import torch
import cv2

def draw_bbox(marks, image_path, output_img, output_tg_img):
    imgcv = cv2.imread(image_path)
    for tensor in marks:
        [x1, y1, x2, y2, _, _] = tensor.tolist()
        cv2.rectangle(imgcv, (round(x1), round(y1)), (round(x2), round(y2)), (0, 0, 255), 1)
    cv2.imwrite(output_img, imgcv)
    cv2.imwrite(output_tg_img, imgcv)
