import requests
import torch
from PIL import Image

from transformers import AutoImageProcessor, SuperPointForKeypointDetection


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open("./data/images.jpeg")

processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint", device_map="auto")

inputs = processor(image, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model(**inputs)

# Post-process to get keypoints, scores, and descriptors
image_size = (image.height, image.width)
outputs = processor.post_process_keypoint_detection(outputs, [image_size])


print(outputs)

import matplotlib.pyplot as plt

plt.axis("off")
plt.imshow(image)
plt.scatter(
    outputs[0]["keypoints"][:, 0],
    outputs[0]["keypoints"][:, 1],
    c=outputs[0]["scores"] * 100,
    s=outputs[0]["scores"] * 50,
    alpha=0.8
)
plt.savefig(f"output_image.png")
