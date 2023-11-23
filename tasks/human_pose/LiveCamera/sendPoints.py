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
import time
import socket
import argparse

parser = argparse.ArgumentParser(description='Send joint coordinates over a network.')
parser.add_argument('--ip', type=str, required=True, help='IP address of the receiving computer.')
parser.add_argument('--port', type=int, required=True, help='Port number for communication.')
args = parser.parse_args()
receiver_ip = args.ip
receiver_port = args.port

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


def get_joint_coordinates(image, counts, objects, peaks):
    joint_coordinates = []

    height, width, _ = image.shape
    K = topology.shape[0]

    count = int(counts[0])
    for i in range(count):
        obj = objects[0][i]
        coordinates = []

        for j in range(len(obj)):
            k = int(obj[j])
            if k >= 0:
                peak = peaks[0][j][k]
                x = round(float(peak[1]) * width)
                y = round(float(peak[0]) * height)
                coordinates.append((x, y))

        joint_coordinates.append(coordinates)

    return joint_coordinates


def execute_frame(image):
    data = preprocess_jpeg(image)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)

    joint_coordinates = get_joint_coordinates(image, counts, objects, peaks)
    
    draw_objects(image, counts, objects, peaks)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_image, joint_coordinates


camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# demonstrate some feature access
new_width = camera.Width.Value - camera.Width.Inc
if new_width >= camera.Width.Min:
    camera.Width.Value = new_width

#cv2.namedWindow("Pose Feed", cv2.WINDOW_NORMAL)

camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

frame_count = 0
start_time = time.time()

# Create a socket object
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the address and port of the receiving computer
receiver_address = (receiver_ip, receiver_port)
sock.connect(receiver_address)

while cv2.waitKey(1) != 27:
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Convert the image to OpenCV format
        img = grabResult.Array
        if img is not None:
            res = execute_frame(img)
            # Send joint coordinates to the receiver
            joint_coordinates_str = json.dumps(res[1])
            try:
                sock.sendall(joint_coordinates_str.encode())
            except BrokenPipeError:
                print("Connection with the receiver is broken!")
                break
            # Print coordinates on stdout
            print(res[1])
            #cv2.imshow("Pose Feed", res[0])
            frame_count += 1

    grabResult.Release()

    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        frame_rate = frame_count / elapsed_time
        print(f"Frame rate: {frame_rate:.2f} frames per second")
        frame_count = 0
        start_time = time.time()

# Release the OpenCV window and close the camera
cv2.destroyAllWindows()
camera.Close()
sock.close()
