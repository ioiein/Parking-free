import os
import numpy as np
import torch

from map_general import create_map

MODEL_PATH = 'model/best.pt'
CONF = 0.4
WIDTH = 1920
HEIGHT = 1080


def main():
    # load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
    model.conf = CONF

    top_errors = []
    right_errors = []
    bottom_errors = []
    left_errors = []
    central_errors = []
    total_errors = []

    # cycle for all images
    for img in os.listdir('images'):
        # make prediction
        result = model(os.path.join('images', img))
        marks = []
        _, predict_space = create_map(result.xywh[0])

        # read marks from file for this image
        with open(os.path.join('labels', img.replace('.jpg', '.txt')), 'r') as fo:
            for line in fo:
                _, x, y, _, _ = line.split()
                marks.append(torch.tensor([float(x) * WIDTH, float(y) * HEIGHT]))
        _, true_space = create_map(marks)

        # append errors for each slot
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

    print('top mae: ', np.abs(top_errors).mean())
    print('right mae: ', np.abs(right_errors).mean())
    print('bottom mae: ', np.abs(bottom_errors).mean())
    print('left mae: ', np.abs(left_errors).mean())
    print('central mae: ', np.abs(central_errors).mean())
    print('total mae: ', np.abs(total_errors).mean())


if __name__ == "__main__":
    main()
