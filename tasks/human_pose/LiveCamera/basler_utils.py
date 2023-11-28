from pypylon import pylon 
from pypylon import genicam
import sys
import cv2
"""
try:
    from .image import *
except:
    from image import *
"""

class frame_extractor:
    def __init__(self):
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
        self.load_devices()


    def load_devices(self):
        self.tlf = pylon.TlFactory.GetInstance()
        self.devices = self.tlf.EnumerateDevices([pylon.DeviceInfo(),])
        self.n_devices = len(self.devices)
        print(f"PYLON: {self.n_devices} devices detected")
        if self.n_devices == 0:
            raise Exception("No devices detected!")

    def start_single_cam(self):
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def start_cams(self, num_cameras:int = None, signal_period:int = 200000, exposure_time:int = 10000):
        # get devices
        if num_cameras is None:
            num_cameras = len(self.devices)
        self.num_cameras = num_cameras
        self.cam_array = pylon.InstantCameraArray(min(len(self.devices), self.num_cameras))
        for i, cam in enumerate(self.cam_array):
            cam.Attach(self.tlf.CreateDevice(self.devices[i]))
            print("Using device ", cam.GetDeviceInfo().GetModelName())
        self.cam_array.Open()

        for idx, cam in enumerate(self.cam_array):
            camera_serial = cam.DeviceInfo.GetSerialNumber()
            cam.SetCameraContext(idx)
            print(f"set context {idx} for camera {camera_serial}")

        # set exposure time
        for camera in self.cam_array:
            camera.ExposureTime.SetValue(exposure_time)

        #set hardware trigger 
        for camera in self.cam_array:
           camera.BslPeriodicSignalPeriod=signal_period
           camera.BslPeriodicSignalDelay=0
           camera.TriggerSelector="FrameStart"
           camera.TriggerMode="On"
           camera.TriggerSource="PeriodicSignal1"
           print(f"Camera {camera.GetDeviceInfo().GetModelName()}: TriggerSource is {camera.TriggerSource.GetValue()}")

        self.cam_array.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    
    def grab_multiple_cams(self, timeout:int = 5000):
        assert(self.cam_array.IsGrabbing())

        images = [None] * self.num_cameras
        while True:
            # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            grabResult = self.cam_array.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)

            # Image grabbed successfully?
            if grabResult.GrabSucceeded():
                # Access the image data.
                cam_id = grabResult.GetCameraContext()
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                grabResult.Release()
                images[cam_id] = Image(img=img)
                if any( im is None for im in images):
                    continue
                else:
                    break
        return images
    
    def grab_multiple_frames(self, timeout:int = 5000):
        """
        Grab multiple frames and returns the img arrays
        """
        assert(self.cam_array.IsGrabbing())

        images = [None] * self.num_cameras
        while True:
            # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
            grabResult = self.cam_array.RetrieveResult(timeout, pylon.TimeoutHandling_ThrowException)

            # Image grabbed successfully?
            if grabResult.GrabSucceeded():
                # Access the image data.
                cam_id = grabResult.GetCameraContext()
                #print("Camera ", cam_id, ": ", self.cam_array[cam_id].GetDeviceInfo().GetSerialNumber())
                image = self.converter.Convert(grabResult)
                img = image.GetArray()
                #print(img)
                grabResult.Release()
                images[cam_id] = img
                if any( im is None for im in images):
                    continue
                else:
                    break
        return images
    

    def grab_one_frame_per_camera(self):
        frames = []
        for cam_id in range(self.num_cameras):
            grab_result = self.cam_array[cam_id].RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                image = self.converter.Convert(grab_result)
                frame = image.GetArray()
                grab_result.Release()
                frames.append(frame)
            else:
                print(f"Error grabbing frame from camera {cam_id}: {grab_result.ErrorCode}, {grab_result.ErrorDescription}")

        return frames


    def collect_frames_multiple(self, manual:bool=True, max_frames:int = 50, show:bool=True, func_show=[], drop_rate:int=2):
        collection = []
        while True:
            rawframes = self.grab_multiple_cams()

            if show:
                imgs_show = [ r for r in rawframes ]
                for func in func_show:
                    imgs_show = func[0](imgs_show, *func[1])
                for cam_id,img_show in enumerate(imgs_show):
                    resized = cv2.resize(img_show.img.numpy(), (int(img_show.img.shape[1]/drop_rate), int(img_show.img.shape[0]/drop_rate)), interpolation= cv2.INTER_LINEAR)
                    cv2.imshow("Cam_"+str(cam_id), resized)
            key = cv2.waitKey(1)
            space_pressed = key == ord(' ')
            if not manual or space_pressed:
                if space_pressed: print("Space pressed")
                print("Frame collected")
                collection.append(rawframes)
                if len(collection)>=max_frames:
                    print(f"Max n frames collected ({max_frames})")
                    break

            elif key == ord('q'):
                print("Q pressed")
                print("Sequence grabbed")
                cv2.destroyAllWindows()
                break

        return collection


        
    def stop_single_cam(self):
        self.camera.Close()

    def stop_multiple_cams(self):
        self.cam_array.Close()

    def grab_single_cam(self):
        assert(self.camera.IsGrabbing())

        # Wait for an image and then retrieve it. A timeout of 5000 ms is used.
        grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        # Image gimg=rabbed successfully?
        if grabResult.GrabSucceeded():
            # Access the image data.
            image = self.converter.Convert(grabResult)
            img = Image(img=image.GetArray())
            grabResult.Release()
            # cv2.imshow( "grabbed", img )
            # cv2.waitKey(0)
            return img
        else:
            print("Error: ", grabResult.ErrorCode, grabResult.ErrorDescription)


if __name__=="__main__":
    frame_extr = frame_extractor()
    frame_extr.start_single_cam()

    # # single frame
    # frame_extr.start_single_cam()
    # img = frame_extr.grab_single_cam()
