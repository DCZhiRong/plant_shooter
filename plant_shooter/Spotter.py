#! /usr/bin/env python3
import os
import busio
import time
from adafruit_servokit import ServoKit
import rclpy # Python Client Library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
from ament_index_python.packages import get_package_share_directory
bringup_dir = get_package_share_directory('plant_shooter')
classNames = []
classFile = os.path.join(bringup_dir, 'Object_Detection_Files', 'coco.names')
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")
configPath = os.path.join(bringup_dir, 'Object_Detection_Files', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
weightsPath = os.path.join(bringup_dir, 'Object_Detection_Files', 'frozen_inference_graph.pb')

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

nbPCAServo=2

MIN_IMP  =[1000, 1000]
MAX_IMP  =[2500, 2500]
MIN_ANG  =[30, 30]
MAX_ANG  =[150, 150]

pca = ServoKit(channels=16, i2c=busio.I2C((2,8),(2,7)))


p = 0.04
i = 0.001
d = 0.0000001

class ImagePublisher(Node):

  def __init__(self):

    super().__init__('image_publisher')

    self.publisher_ = self.create_publisher(Image, 'video_frames', 10)

    timer_period = 0.03  # seconds

    self.timer = self.create_timer(timer_period, self.timer_callback)
         
    self.cap = cv2.VideoCapture(0)
    #self.cap.set(3,640)
    #self.cap.set(4,480)
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()

    self.prevT = time.time()
    self.ex = 0
    self.ey = 0
    self.prex = 0
    self.prey = 0


    for i in range(nbPCAServo):
        pca.servo[i].set_pulse_width_range(MIN_IMP[i] , MAX_IMP[i])
        pca.servo[i].angle = 90

   
  def timer_callback(self):
    ret, frame = self.cap.read()     
    frame = cv2.flip(frame,0)  
    result, objectInfo = self.getObjects(frame,0.5,0.2, objects=['cell phone'])
    curT = time.time()
    # Publish the image.
    # The 'cv2_to_imgmsg' method converts an OpenCV
    # image to a ROS 2 image message
    self.publisher_.publish(self.br.cv2_to_imgmsg(frame, 'bgr8'))
    if objectInfo:
      x_error = 640-(objectInfo[0][0][0]+objectInfo[0][0][2]/2)
      y_error = 360-(objectInfo[0][0][1]+objectInfo[0][0][3]/2)
      time_diff = curT-self.prevT
      self.preT = curT
      dedtX = (x_error - self.prex)/time_diff
      dedtY = (y_error - self.prey)/time_diff
      self.ex += x_error*time_diff
      self.ey += y_error*time_diff
      self.prex = x_error
      self.prey = y_error
      ux = 0.5*p*x_error + 0.5*i*self.ex + d*dedtX
      uy = p*y_error + i*self.ey + d*dedtY
      ux = min(60, max(-60, ux))
      uy = min(60, max(-60, uy))
      #print(ux)
      pca.servo[0].angle = -uy+90
      pca.servo[1].angle = ux+90

      
    # Display the message on the console
    # self.get_logger().info('Publishing video frame')

  def getObjects(self, img, thres, nms, draw=True, objects=[]):
      classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
  #Below has been commented out, if you want to print each sighting of an object to the console you can uncomment below     
      if len(objects) == 0: objects = classNames
      objectInfo =[]
      if len(classIds) != 0:
          for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
              className = classNames[classId - 1]
              if className in objects: 
                  objectInfo.append([box,className])
                  if (draw):
                      cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                      cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                      cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                      cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                      cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
      
      return img,objectInfo

def main(args=None):
  rclpy.init(args=args)
  image_publisher = ImagePublisher()
  # Spin the node so the callback function is called.
  rclpy.spin(image_publisher)
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_publisher.destroy_node()
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()