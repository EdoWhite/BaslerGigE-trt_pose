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
import argparse
from basler_utils import frame_extractor

parser = argparse.ArgumentParser(description='3D pose from 2D')
parser.add_argument('--calibration_matrix', type=str, required=True, help='')
args = parser.parse_args()
calibration = args.calibration_matrix

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
    #K = topology.shape[0]

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

def get_calib_parameters():
    fs = cv2.FileStorage(calibration, cv2.FILE_STORAGE_READ)
    n_cams = int(fs.getNode("nb_camera").real())
    calib_data = {}
    for i in range(n_cams):
        name = "camera_"+str(i)
        cam = fs.getNode( name )
        calib_data[name]= {
                    "camera_matrix" : cam.getNode("camera_matrix").mat(),
                    "distortion_vector" : cam.getNode("distortion_vector").mat(),
                    "camera_pose_matrix" : cam.getNode("camera_pose_matrix").mat(),
                    "img_width" : cam.getNode("img_width").real(),
                    "img_height" : cam.getNode("img_height").real()
                }
    fs.release()
    return calib_data

frame_extr = frame_extractor()
frame_extr.start_cams()

while cv2.waitKey(1) != 27:
    #Get Calib Matrices
    calib = get_calib_parameters()
    """
    for index, (camera_name, camera_data) in enumerate(calib.items()):
        print(f"  Camera {index} ({camera_name}):")
        print(f"  Camera Matrix:\n{camera_data['camera_matrix']}")
        print(f"  Distortion Vector:\n{camera_data['distortion_vector']}")
        print(f"  Camera Pose Matrix:\n{camera_data['camera_pose_matrix']}")
        print(f"  Image Width: {camera_data['img_width']}")
        print(f"  Image Height: {camera_data['img_height']}")
        print("\n")
    """
    # Get Frames
    #frames = frame_extr.grab_one_frame_per_camera()
    frames = frame_extr.grab_multiple_frames()

    # Get 2D coordinates
    joints = []
    for index, frame in enumerate(frames):
        coord = execute_frame(frame)[1]
        joints.append(coord)
        print(f"Coordinates frame {index}: {coord}")

    # Triangulate


frame_extr.stop_multiple_cams()

