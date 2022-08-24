import argparse, os, sys
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from yolov7.yolov7_runner import Yolov7Runner
from TrOCR.trocr_runner import TrOCRRunner
from yolov7.utils.general import increment_path

def main(opt):
    # yolov7 init
    yolov7 = Yolov7Runner(opt)
    # Detect text
    all_bboxes = yolov7.run()

    os.chdir('../')
    # Recognize text
    trocr = TrOCRRunner(opt)
    all_labels = trocr.run(all_bboxes)

    # Visualize
    save_result(opt, all_labels)

def save_result(opt, all_labels):
    save_dir = Path(increment_path(Path(opt.name), exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels').mkdir(parents=True, exist_ok=True)  # make dir

    # Save json
    with open(save_dir / "labels.json", 'w', encoding='utf-8') as f:
        json.dump(all_labels, f)

    for img_name in all_labels:
        img_path = Path(opt.source) / img_name
        image = cv2.imread(str(img_path))

        for label in all_labels[img_name]:
            category = label['category']
            category_id = label['category_id']
            text = label['text']
            det_conf = label['det_conf']
            x_min = label['x_min']
            y_min = label['y_min']
            x_max = label['x_max']
            y_max = label['y_max']

            points = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            if   category_id == 0: color = (0, 255, 0)
            elif category_id == 1: color = (0, 0, 255)
            elif category_id == 2: color = (255, 0, 0)
            elif category_id == 3: color = (255, 255, 0)
            elif category_id == 4: color = (0, 255, 255)
            else:                  color = (255, 0, 255)

            image = cv2.polylines(image, [np.array(points).astype(int)], True, color, 2)
            
            if opt.font != '':
                image_pil = Image.fromarray(image)
                draw = ImageDraw.Draw(image_pil)
                draw.text((points[0][0] - 20, points[0][1] - 20), text, 
                        font=ImageFont.truetype(str(Path(opt.font)), 32), fill=color)
                image = np.array(image_pil)
        cv2.imwrite(str(save_dir/img_name), image)
        



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # yolov7 config
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')

    parser.add_argument('--output', type=str, default='output', help='output folder')
    parser.add_argument('--font', type=str, default='', help='font path')
    parser.add_argument('--recog_model', type=str, default='', help='Recognition model path')

    opt = parser.parse_args()
    # print(opt)

    main(opt)