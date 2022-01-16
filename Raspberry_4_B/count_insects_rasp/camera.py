from picamera import PiCamera
import cv2
import time
from datetime import datetime
import count_insects_rasp.init as init


def create_name_of_image(count,predict):
    now=datetime.now()
    timestr = now.strftime("%Y%m%d%H%M%S")
    imageFileName = f"f_{timestr}_{count}_{predict}.jpg"
    return imageFileName
    

def save_image(image_file_name,image,jpeg_quality=90):
    try:
        cv2.imwrite(image_file_name, image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]) #SELECT 0-100PRI
    except:
        init.my_logs.info(f'Can not save image {image_file_name}')
    


def capture_image():
    starttime=datetime.now()
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        init.my_logs.info("Cannot open camera")
        return None
    ret,frame=cap.read()
    if(frame is not None):
        init.my_logs.info(f'Image capture')
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame=cv2.resize(frame,(240,240),interpolation= cv2.INTER_LINEAR)
        endtime=datetime.now()
        init.my_logs.info(f'Capture time {(endtime-starttime).microseconds} microseconds')
        return frame
        
  