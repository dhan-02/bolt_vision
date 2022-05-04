# bolt_vision

Bounding boxes messages are published in "/bb_detect" topic, and the message type is [Detection2D](http://docs.ros.org/en/api/vision_msgs/html/msg/Detection2D.html). 

Segmented images are published in "/segmented_object_image" topic.

Class-ID of detected objects:
---

* 0  - tire
* 1  -  safety_vest
* 2  -  Contruction_barrel
* 3  -  Stop sign
* 4  -  Pedestrian



To launch the package:
--
```
#run
roslaunch bolt_vision bb_publisher.launch                  #if segmentation is not needed

roslaunch bolt_vision bb_publisher.launch segmentation:=1  #if segmentation is needed

```

Additional Package needed :
--
```
# if torch is not installed, also if cuda is installed previously, check pytorch compatability
conda install pytorch==1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

#For OpenCV
pip install opencv-python


```