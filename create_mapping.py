from PIL import Image, ImageFilter, ImageOps, ImageDraw
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
import os 
from utils.fs import get_file_name

image_path = "./imgs/office_grayscale.jpg"
image_name = get_file_name(image_path)

greyscale_image = Image.open(image_path)

# Apply a Gaussian blur to the binary mask to smooth the boundaries
greyscale_image = greyscale_image.point(lambda p: p - 10 if p > 127.5 else p + 75)
smoothed_mask = greyscale_image.filter(ImageFilter.BoxBlur(5))  # Adjust radius as needed

# Paste the smoothed mask onto a black background to create the smoothed image
smoothed_image = Image.new("L", greyscale_image.size, color=0)
smoothed_image.paste(smoothed_mask,(0, 0))
# smoothed_image = smoothed_image.point(lambda p: p - 20 if p > 127.5 else p + 20)

# Save or display the smoothed image
smoothed_image.save(f"./imgs/{image_name}_mapping.jpg")