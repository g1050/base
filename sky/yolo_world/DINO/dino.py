import os
import torch

# settings
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)
HOME = os.getcwd()
print(HOME)

CONFIG_PATH = "D:\/buaa\/base\sky\yolo_world\DINO\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
WEIGHTS_PATH = WEIGHTS_NAME
print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))

# %cd {HOME}

from groundingdino.util.inference import load_model, load_image, predict, annotate

model = load_model(CONFIG_PATH, WEIGHTS_PATH)

# 下载示例图片
# %cd {HOME}
# !mkdir {HOME}/data
# %cd {HOME}/data

# !wget -q https://media.roboflow.com/notebooks/examples/dog.jpeg
# !wget -q https://media.roboflow.com/notebooks/examples/dog-2.jpeg
# !wget -q https://media.roboflow.com/notebooks/examples/dog-3.jpeg
# !wget -q https://media.roboflow.com/notebooks/examples/dog-4.jpeg

import supervision as sv

IMAGE_NAME = "person.jpg"
IMAGE_PATH = os.path.join("../yolo/img", IMAGE_NAME)
TEXT_PROMPT = "Rottweiler" # 可以换成"dog"，则只给出dog
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model, 
    image=image, 
    caption=TEXT_PROMPT, 
    box_threshold=BOX_TRESHOLD, 
    text_threshold=TEXT_TRESHOLD
)

annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

# %matplotlib inline  
sv.plot_image(annotated_frame, (16, 16))