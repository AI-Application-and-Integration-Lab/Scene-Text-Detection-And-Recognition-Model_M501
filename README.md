# Scene Text Detection and Recognition model

## Installation
``` bash
# create conda environment
conda create -n STDR python=3.8 -y
conda activate STDR

##  Follow yolov7 installation
# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# install pytorch
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# install yolov7 requirement
pip install -r yolov7/requirements.txt

# install transformers
pip install transformers

```

## Data preparation
Download the dataset, and place them into the `dataset` directory.
```
.
├── yolov7
├── TrOCR
├── dataset
│   ├── D501
│   │   ├── train
│   │   │   ├── images
│   │   │   └── labels
│   │   ├── val
│   │   └── test
│   └── ...
└── ...
```
You can also use the AICUP competition dataset for training and testing. See more: [Link](https://tbrain.trendmicro.com.tw/Competitions/Details/19)

For the D501 dataset, run the following command to convert the labels to YOLO format:
```bash
python util/transform_data_D501toYolo.py
```

For the AICUP competition dataset, run the following command to convert the labels to YOLO format:
```bash
python util/transform_data_AICUPtoYolo.py
```


## Testing
[`last.pt`](https://drive.google.com/file/d/1et_BXXtgXhsm-uQFXiZy-6TR7BXc4tMf/view?usp=sharing)

Download the file and place it into `yolov7` folder.

### Detection Module
Since we use YOLOv7 as our detection module, follow the section **Testing** in the YOLOv7 README.
``` bash
cd yolov7
python test.py --data data/D501_Str.yaml --img 1280 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights last.pt --name D501_test --task test
```

### Recognition Module
See [this notebook](TrOCR/train.ipynb)

## Training
### Detection Module
Follow the section **Training** in the YOLOv7 README.
``` bash
cd yolov7
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_aux.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 128 --data data/D501_Str.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name D501_train --hyp data/hyp.scratch.p6.yaml
```

### Recognition Module
See [this notebook](TrOCR/train.ipynb)

## Inference
Please prepare a font file for visualization, for instance, [`Noto Sans Traditional Chinese`](https://fonts.google.com/noto/specimen/Noto+Sans+TC) released by Google. Then run the following command.

``` bash
python predict.py --weights yolov7/last.pt --recog_model ycchen/TrOCR-base-ver021-v1 --source <PATH_TO_IMG_OR_FOLDER> --nosave --save-conf --font <PATH_TO_FONT_FILE> --name D501_predict
```

## Results
In the following section, we split AICUP's original training set into a training set(the first 14,188 images) and a testing set(the last 1,000 images) for training and testing.
When training the AICUP competition dataset, you can use our D501 weight as pre-train weights, and get better performance than training from scratch(using yolov7 default pre-train).
The following measures are expressed as percentages. We only use string categories for training and testing.

### Detection
|    Train    |  Finetune   |  Testing   |  Precision |   Recall   |  F1 score  |
|-------------|-------------|------------|------------|------------|------------|
| D501_train  |      -      | D501_val   |    95.1    |    83.6    |    89.0    |
| D501_train  |      -      | D501_test  |    94.5    |    86.2    |    90.2    |
| AICUP_train |      -      | AICUP_test |    80.7    |    77.6    |    79.1    |
| D501_train  | AICUP_train | AICUP_test | 81.3(+0.6) | 78.6(+1.0) | 79.9(+0.8)   |

### Recognition
|    Train    |  Finetune   |  Testing   |  CER (Character Error Rate) |
|-------------|-------------|------------|-----------------------------|
| D501_train  |      -      | D501_val   |    9.1                      |
| D501_train  |      -      | D501_test  |   11.08                     |
| AICUP_train |      -      | AICUP_test |   14.78                     |
| D501_train  | AICUP_train | AICUP_test |    9.29(-5.49)              |

## Reference

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
@misc{li2021trocr,
      title={TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models}, 
      author={Minghao Li and Tengchao Lv and Lei Cui and Yijuan Lu and Dinei Florencio and Cha Zhang and Zhoujun Li and Furu Wei},
      year={2021},
      eprint={2109.10282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
