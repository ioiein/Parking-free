import os
import numpy as np
import torch

from map_general import create_map

MODEL_PATH = 'model/best.pt'
WIDTH = 1920
HEIGHT = 1080


def main():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
    model.conf = 0.4
    top_errors = []
    right_errors = []
    bottom_errors = []
    left_errors = []
    central_errors = []
    total_errors = []
    for img in os.listdir('images'):
        result = model(os.path.join('images', img))
        marks = []
        _, predict_space = create_map(result.xywh[0])
        with open(os.path.join('labels', img.replace('.jpg', '.txt')), 'r') as fo:
            for line in fo:
                _, x, y, _, _ = line.split()
                marks.append(torch.tensor([float(x) * WIDTH, float(y) * HEIGHT]))
        _, true_space = create_map(marks)
        top_errors.append(predict_space['top'] - true_space['top'])
        right_errors.append(predict_space['right'] - true_space['right'])
        bottom_errors.append(predict_space['bottom'] - true_space['bottom'])
        left_errors.append(predict_space['left'] - true_space['left'])
        central_errors.append(predict_space['central'] - true_space['central'])
        total_errors.append(predict_space['top'] - true_space['top'] +
                            predict_space['right'] - true_space['right'] +
                            predict_space['bottom'] - true_space['bottom'] +
                            predict_space['left'] - true_space['left'] +
                            predict_space['central'] - true_space['central'])

    print('top mse: ', np.square(top_errors).mean())
    print('right mse: ', np.square(right_errors).mean())
    print('bottom mse: ', np.square(bottom_errors).mean())
    print('left mse: ', np.square(left_errors).mean())
    print('central mse: ', np.square(central_errors).mean())
    print('total mse: ', np.square(total_errors).mean())

    print('top mae: ', np.abs(top_errors).mean())
    print('right mae: ', np.abs(right_errors).mean())
    print('bottom mae: ', np.abs(bottom_errors).mean())
    print('left mae: ', np.abs(left_errors).mean())
    print('central mae: ', np.abs(central_errors).mean())
    print('total mae: ', np.abs(total_errors).mean())


if __name__ == "__main__":
    main()
