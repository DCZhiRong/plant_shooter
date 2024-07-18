#! /usr/bin/env python3
#import mraa 
import os
import busio
import time
from adafruit_servokit import ServoKit
import rclpy # Python Client Library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from geometry_msgs.msg import Twist
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

MIN_IMP  =[1000, 1000, 500]
MAX_IMP  =[2500, 2500, 2000]
MIN_ANG  =[30, 30, 0]
MAX_ANG  =[150, 150, 180]

pca = ServoKit(channels=16, i2c=busio.I2C((2,8),(2,7)))


p = 0.04
i = 0.0002
d = 0.000000

class ImagePublisher(Node):

  def __init__(self):

    super().__init__('image_publisher')

    self.publisher_ = self.create_publisher(Image, 'video_frames', 10)

    timer_period = 1/15  # seconds

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

    self.tarx = 0
    self.tary = 0
    self.angx = 0
    self.angy = 0

    self.servx = 90
    self.servy = 90

    for i in range(nbPCAServo):
      pca.servo[i].set_pulse_width_range(MIN_IMP[i] , MAX_IMP[i])
      pca.servo[i].angle = 90
    pca.servo[2].set_pulse_width_range(MIN_IMP[2] , MAX_IMP[2])
    pca.servo[2].angle = 0
    
    self.subscription1 = self.create_subscription(Twist, 'cmd_vel', self.listener_callback1, 10)

  def listener_callback1(self, msg):
    self.servx += msg.angular.z
    self.servy -= msg.linear.x*2
    self.servx = min(150, max(30, self.servx))
    self.servy = min(150, max(30, self.servy))
    tar_angle = pca.servo[2].angle
    if msg.linear.z > 0:
      # print('high')
      tar_angle = 180
    elif msg.linear.z < 0:
      # print('low')
      tar_angle = 0
    while int(pca.servo[2].angle - tar_angle) != 0:
      pca.servo[2].angle = pca.servo[2].angle*0.9 + tar_angle*0.1

   
  def timer_callback(self):
    ret, frame = self.cap.read()     
    frame = cv2.flip(frame,0)  
    result, objectInfo = self.getObjects(frame,0.6,0.2, objects=['bottle'])
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
      self.tarx = min(150-self.servx, max(30-self.servx, ux))
      self.tary = min(150-self.servy, max(30-self.servy, uy))
      #print(ux)
    if self.tarx + self.servx > 150 or self.tarx + self.servx < 60:
      self.tarx = min(150-self.servx, max(30-self.servx, self.tarx))
    if self.tary + self.servy > 150 or self.tarx + self.servy < 60:
      self.tary = min(150-self.servy, max(30-self.servy, self.tary))
    self.angx = self.angx*0.6 + self.tarx*0.4
    self.angy = self.angy*0.6 + self.tary*0.4
    pca.servo[0].angle = -self.angy+self.servy
    pca.servo[1].angle = self.angx+self.servx

      
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