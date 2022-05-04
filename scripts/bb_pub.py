#!/usr/bin/env python

import rospy
import cv2
import torch
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge, CvBridgeError
#import numpy as np
bridge = CvBridge()


def bb_detect():
    rospy.init_node('bounding_box', anonymous=True)
    global pub, model, output_pub
    model_path = rospy.get_param("/bb_publisher/model_path")
    pub = rospy.Publisher("/bb_detect", Detection2D, queue_size=20)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
    image_sub = rospy.Subscriber("/zed2i/zed_node/rgb/image_rect_color", Image, image_callback)
    output_pub = rospy.Publisher("/bounding_box_output", Image, queue_size=20)
    rospy.loginfo("Waiting for image topics...")
    rospy.spin()


def image_callback(data):
    try:
        global bridge
        image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        print(e)
    global model

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    results = model(img, size=640)
    pred = results.xyxy[0].cpu().detach().numpy()
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

        pub.publish(bb_detect)
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)
        # const auto acc_img = cv_bridge::CvImage({}, sensor_msgs::image_encodings::MONO8, accumulator_img);
        # cv_pub4.publish(acc_img.toImageMsg());
    try:
        output_pub.publish(bridge.cv2_to_imgmsg(image, "bgr8"))
    except CvBridgeError as e:
        print(e)

if(__name__ == "__main__"):
    bb_detect()


