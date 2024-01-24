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
import argparse

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
    #rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--frame_interval', type=int, required=False, default=1)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    input_video_path = args.video_path
    frame_interval = args.frame_interval
    #OPTIMIZED_MODEL = '../densenet121_baseline_att_256x256_trt.pth'
    OPTIMIZED_MODEL = args.model_path

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

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)

    cv2.namedWindow("PoseFeed", cv2.WINDOW_NORMAL)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps}")
    print(f"Total Frames: {total_frames}")

    # Loop through the frames and read them as JPG images
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Finished reading the video or error in frame reading.")
            break

        # Save frames at the specified interval
        if frame_count % frame_interval == 0:
            #print("Executing Frame", frame_count)
            res = execute_frame(frame)
            cv2.imshow("PoseFeed", res)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    # Release the video capture object
    cap.release()

    print("Frames extraction completed.")

    # Release the OpenCV 
    cv2.destroyAllWindows()

