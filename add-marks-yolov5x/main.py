import os
import torch


def get_iou(bb1, bb2):
    # determine the coordinates of the intersection rectangle
    bb1_x1 = bb1[1] - bb1[3] / 2.0
    bb1_x2 = bb1[1] + bb1[3] / 2.0
    bb1_y1 = bb1[2] - bb1[4] / 2.0
    bb1_y2 = bb1[2] + bb1[4] / 2.0
    bb2_x1 = float(bb2[1]) - float(bb2[3]) / 2.0
    bb2_x2 = float(bb2[1]) + float(bb2[3]) / 2.0
    bb2_y1 = float(bb2[2]) - float(bb2[4]) / 2.0
    bb2_y2 = float(bb2[2]) + float(bb2[4]) / 2.0
    x_left = max(bb1_x1, bb2_x1)
    y_top = max(bb1_y1, bb2_y1)
    x_right = min(bb1_x2, bb2_x2)
    y_bottom = min(bb1_y2, bb2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1_x2 - bb1_x1) * (bb1_y2 - bb1_y1)
    bb2_area = (bb2_x2 - bb2_x1) * (bb2_y2 - bb2_y1)

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def add_marks():
    # load model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

    # cycle for all images in PKLot
    for cam in os.scandir('images'):
        for weather in os.scandir(cam):
            for file in os.listdir(weather):
                detections = []
                detections.clear()
                marks = []
                marks.clear()
                # make prediction by model
                result = model(os.path.join(weather, file))
                result_tensor = result.xywhn[0]
                for tensor in result_tensor:
                    [x, y, w, h, _, cl] = tensor.tolist()
                    # 2 - car, 7 - truck
                    if cl == 2 or cl == 7:
                        detections.append([0, x, y, w, h])

                # read bbox from marks of PKLot
                with open(os.path.join('labels', cam.name, weather.name, file.replace('.jpg', '.txt')), 'r') as fo:
                    for line in fo:
                        marks.append(line.split())

                added_marks = []
                added_marks.clear()
                # compare bbox in detections from model with marks from PKLot
                for det in detections:
                    no_intersection = True
                    det_str = [str(i) for i in det]
                    for mark in marks:
                        iou = get_iou(det, mark)
                        if iou > 0:
                            no_intersection = False
                        # bbox from model better > so replace
                        if iou > 0.35:
                            added_marks.append(det_str)
                            marks.remove(mark)
                            break
                    if no_intersection:
                        added_marks.append(det_str)
                marks.extend(added_marks)
                with open(os.path.join('labels', file.replace('.jpg', '.txt')), 'w') as fw:
                    for mark in marks:
                        s = " ".join(mark) + '\n'
                        fw.write(s)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    add_marks()
