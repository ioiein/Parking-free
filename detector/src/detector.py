import logging
import logging.config
import yaml
import click
import torch
import json
import cv2
import os
import time

from src.map import create_map


LOGGER_CFG = "configs/logging.conf.yaml"
APPLICATION_NAME = "detector"
MODEL_PATH = "model/last.pt"
CONF_LEVEL = 0.4
IMG_PATH = "data/input/img.jpg"
JSON_PATH = "data/output/out.json"
OUTPUT_IMG = "../server/img/out.jpg"
OUTPUT_TG_IMG = "../telegram_bot/out.jpg"
OUTPUT_MAP = "../server/img/map.jpg"
OUTPUT_TG_MAP = "../telegram_bot/map.jpg"
OUTPUT_FREE = "../server/img/slots.json"
OUTPUT_TG_FREE = "../telegram_bot/slots.json"

logger = logging.getLogger(APPLICATION_NAME)
global model, results


def setup_logging(path: str) -> None:
    with open(path) as config_f:
        logging.config.dictConfig(yaml.safe_load(config_f))


def draw_bbox(marks, image_path=IMG_PATH, output_img=OUTPUT_IMG, output_tg_img=OUTPUT_TG_IMG):
    imgcv = cv2.imread(image_path)
    for tensor in marks:
        [x1, y1, x2, y2, _, _] = tensor.tolist()
        cv2.rectangle(imgcv, (round(x1), round(y1)), (round(x2), round(y2)), (0, 0, 255), 1)
    cv2.imwrite(output_img, imgcv)
    cv2.imwrite(output_tg_img, imgcv)


@click.command(name="detect")
def detect_command():
    setup_logging(LOGGER_CFG)
    logger.info(f"script started")

    # load model
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH)
        logger.info(f"model loaded")
    except Exception as err:
        logger.error(f"model didnt load, {err} happened")
        exit(-1)
    # set conf
    model.conf = CONF_LEVEL
    logger.info(f"confidence threshold set")

    while True:
        logger.info(f"iteration started")
        # dowload image from stream
        os.system(
            f"ffmpeg -y -i https://msk.rtsp.me/XEmxGcyEbsWZaHxQlTe5-w/1635357896/hls/ZdG5F8D5.m3u8 -frames:v 1 {IMG_PATH}"
        )

        # make prediction
        try:
            results = model(IMG_PATH)
            logger.info(f"detection done")
        except Exception as err:
            logger.error(f"{err} happened")

        # drow box, make map and save slots dict
        draw_bbox(results.xyxy[0])
        parking_map, free_slots = create_map(results.xywh[0])
        cv2.imwrite(OUTPUT_MAP, parking_map)
        cv2.imwrite(OUTPUT_TG_MAP, parking_map)
        with open(OUTPUT_FREE, 'w') as f:
            json.dump(free_slots, f)
        with open(OUTPUT_TG_FREE, 'w') as f:
            json.dump(free_slots, f)
        with open(JSON_PATH, 'w') as fi:
            json.dump(results.pandas().xywhn[0].to_json(), fi)
        logger.info(f"json saved")
        logger.info(f"iteration ended\n")
        time.sleep(60)


if __name__ == "__main__":
    detect_command()
