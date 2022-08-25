from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
from pathlib import Path
from tqdm import tqdm

class TrOCRRunner:
    def __init__(self, opt):
        self.opt = opt
        self.processor = TrOCRProcessor.from_pretrained(self.opt.recog_model)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.opt.recog_model)
    
    def run(self, all_bboxes):
        print("Run Recognition...")
        all_labels = dict()
        source = self.opt.source

        for img_name in tqdm(all_bboxes):
            source_path = Path(source)
            if source_path.is_file():
                img_path = source_path
            else:
                img_path = source_path / img_name
            # Read image
            img = Image.open(img_path)
            w, h = img.size

            labels = []
            for bbox in all_bboxes[img_name]:
                category, category_id, x_center, y_center, width, height \
                    = bbox[0], int(bbox[1]), float(bbox[2]), \
                        float(bbox[3]), float(bbox[4]), float(bbox[5])
                if len(bbox) == 7:
                    conf = float(bbox[6])
                else:
                    conf = None
                
                x_center *= w
                y_center *= h
                width *= w
                height *= h

                x_min = max(0, int(round(x_center - (width / 2))))
                x_max = min(w, int(round(x_center + (width / 2))))
                y_min = max(0, int(round(y_center - (height / 2))))
                y_max = min(h, int(round(y_center + (height / 2))))

                # Crop image
                crop_img = img.crop((x_min, y_min, x_max, y_max))

                # Processor
                pixel_values = self.processor(images=crop_img, return_tensors="pt").pixel_values
                generated_ids = self.model.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                labels.append(dict(category = category,
                                    category_id = category_id,
                                    x_min = x_min,
                                    y_min = y_min,
                                    x_max = x_max,
                                    y_max = y_max,
                                    det_conf = conf,
                                    text = generated_text))
            all_labels[img_name] = labels
        return all_labels