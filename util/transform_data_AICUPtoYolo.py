import os
import json
from tqdm import tqdm

gts_root = "../dataset/AICUP/"  # Source
output = "../dataset/AICUP/"    # Target

for split_name in ['train', 'val', 'test']:
    output_path = os.path.join(output, split_name)

    gts_path = os.path.join(gts_root, split_name, "labels")
    gts = sorted(os.listdir(gts_path))
    for gt_file in tqdm(gts):
        with open(os.path.join(gts_path, gt_file), 'r', encoding='utf-8') as gt_f:
            data = json.load(gt_f)
            img_shape = [int(data['imageHeight']), int(data['imageWidth'])]

            txt = []
            for shape in data['shapes']:
                if shape['group_id'] in [0, 2, 3, 4]:
                    group_id = shape['group_id']
                    points = shape['points']

                    # Transform points to yolor format
                    # reference: https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
                    min_x = max(0, min(points[0][0], points[1][0], points[2][0], points[3][0]))
                    min_y = max(0, min(points[0][1], points[1][1], points[2][1], points[3][1]))
                    max_x = min(img_shape[1], max(points[0][0], points[1][0], points[2][0], points[3][0]))
                    max_y = min(img_shape[0], max(points[0][1], points[1][1], points[2][1], points[3][1]))

                    x_center = (max_x+min_x)/2
                    y_center = (max_y+min_y)/2
                    width = max_x - min_x
                    height = max_y - min_y

                    x_center /= img_shape[1]
                    y_center /= img_shape[0]
                    width /= img_shape[1]
                    height /= img_shape[0]

                    if group_id == 2:
                        group_id = 1
                    elif group_id == 3:
                        group_id = 2
                    elif group_id == 4:
                        group_id = 3

                    txt.append([group_id, x_center, y_center, width, height])
            
            with open(os.path.join(output_path, "labels", gt_file.split(".json")[0]+".txt"), 'w') as f:
                for t in txt:
                    f.write(f"{t[0]} {t[1]} {t[2]} {t[3]} {t[4]}\n")
