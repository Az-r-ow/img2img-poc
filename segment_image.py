from PIL import Image, ImageFilter, ImageOps, ImageDraw
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
from utils.fs import get_file_name
import os 

model_name = "facebook/maskformer-swin-large-ade"
segment = pipeline("image-segmentation", model=model_name)

image_path = './imgs/office.jpg'
image = Image.open(image_path)
pred = segment(image)

for i in range(len(pred)):
  if pred[i]['label'] == 'wall':
    inverted_image = ImageOps.invert(pred[i]['mask'])
    greyscale_image = inverted_image.convert("LA").convert("RGB")
    imagename = get_file_name(image_path)
    greyscale_image.save(f"./imgs/{imagename}_grayscale.jpg")