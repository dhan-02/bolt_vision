#!/usr/bin/env python

import rospy
import cv2
import torch
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
bridge = CvBridge()

#class Id of objects which require segmentation
tire_id = 0
safety_vest_id = 1

def bb_detect():
    rospy.init_node('bounding_box', anonymous=True)
    global pub1, pub2,  model
    model_path = rospy.get_param("/bb_publisher/model_path")
    pub1 = rospy.Publisher("/bb_detect", Detection2D, queue_size=20)
    pub2 =  rospy.Publisher("/segmented_object_image", Image, queue_size=20)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    image_sub = rospy.Subscriber("/camera/front/image_raw", Image, image_callback)

    rospy.loginfo("Waiting for image topics...")
    rospy.spin()


def image_callback(data):
    try:
        global bridge
        image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    global model
    ####
    frame = image
    background = np.zeros_like(frame)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    results = model(img, size=640)
    pred = results.xyxy[0].cpu().detach().numpy()
    #print(pred)
    for i in range(len(pred)):
        bb_detect = Detection2D()
        x1, y1, x2, y2, conf, class_ = pred[i][0], pred[i][1], pred[i][2], pred[i][3], pred[i][4], pred[i][5]
        bb_detect.bbox.center.x = int((x1 + x2) / 2)
        bb_detect.bbox.center.y = int((y1 + y2) / 2)
        bb_detect.bbox.size_x = int(x2 - x1)
        bb_detect.bbox.size_y = int(y2 - y1)
        class_conf = ObjectHypothesisWithPose()
        class_conf.id = int(class_)
        class_conf.score = float(conf)
        bb_detect.results = [class_conf]

        pub1.publish(bb_detect)

    if(tire_id in pred[:,5]):
        index = np.where(pred[:,5] == tire_id)[0]
        for id in index:
            x1, y1, x2, y2, conf, class_ = int(pred[id][0]), int(pred[id][1]), int(pred[id][2]), int(pred[id][3]), pred[id][4], pred[id][5]
            dframe = frame[y1:y2, x1:x2]
            mask = np.zeros(dframe.shape[:2], np.uint8)

            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            rect = (1, 1, dframe.shape[1]-2, dframe.shape[0]-2)  # (x, y, w, h)
            cv2.grabCut(dframe, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)                      

            mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1).astype(np.uint8)
            dframe = cv2.bitwise_and(dframe, dframe, mask=mask)
            background[y1:y2, x1:x2] = dframe

    elif(safety_vest_id in pred[:,5]):
        index = np.where(pred[:,5] == safety_vest_id)[0]
        for id in index:
            x1, y1, x2, y2, conf, class_ = int(pred[id][0]), int(pred[id][1]), int(pred[id][2]), int(pred[id][3]), pred[id][4], pred[id][5]
            dframe = frame[y1:y2, x1:x2]
            hsv = cv2.cvtColor(dframe, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv,(2, 100, 100), (20, 255, 255))
            dframe = cv2.bitwise_and(dframe, dframe, mask=mask)
            background[y1:y2, x1:x2] = dframe

    img_msg = bridge.cv2_to_imgmsg(background, "bgr8")
    pub2.publish(img_msg)

if(__name__ == "__main__"):
    bb_detect()


