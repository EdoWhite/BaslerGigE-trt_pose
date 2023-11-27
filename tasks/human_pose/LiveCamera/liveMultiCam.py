from pypylon import pylon
import cv2
import json
import trt_pose.coco
import trt_pose.models
import torch
from torch2trt import TRTModule
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import cv2
import torchvision.transforms as transforms
import PIL.Image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from basler_utils import frame_extractor

WIDTH = 256
HEIGHT = 256
data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')

with open('../human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)

num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

OPTIMIZED_MODEL = '../densenet121_baseline_att_256x256_trt.pth'
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)

def preprocess_jpeg(image):
    global device
    device = torch.device('cuda')
    image = cv2.resize(image, (WIDTH, HEIGHT))
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def execute_frame(image):
    data = preprocess_jpeg(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(image, counts, objects, peaks)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image

def preprocess_jpeg_batch(images):
    global device
    device = torch.device('cuda')
    resized_images = [cv2.resize(image, (WIDTH, HEIGHT)) for image in images]
    tensor_images = [transforms.functional.to_tensor(image).to(device) for image in resized_images]
    tensor_images = torch.stack(tensor_images)
    tensor_images.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor_images

def execute_frame_batch(images):
    res = []
    data = preprocess_jpeg_batch(images)
    for d, i in zip(data, images):
        cmap, paf = model_trt(d)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = parse_objects(cmap, paf)
        draw_objects(i, counts, objects, peaks)
        rgb_image = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        res.append(rgb_image)
    return res

frame_extr = frame_extractor()
frame_extr.start_cams()

cv2.namedWindow("Camera1", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera2", cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27:
    frames = frame_extr.grab_multiple_frames()
    #res1, res2 = execute_frame_batch(frames)
    for index, frame in enumerate(frames):
        cv2.imshow(f"Camera{index+1}", execute_frame(frame))

# Release the OpenCV window and close the camera
cv2.destroyAllWindows()
