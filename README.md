# MotionVision: Object Detection and Tracking with OpenCV and CNN

## Task One: Moving Object Detection

This function performs moving object detection in a video using background subtraction. It uses the OpenCV library to create a background subtractor, which helps identify foreground objects in each frame. The detected objects are classified into three categories: persons, cars, and others. The output includes a visual representation of the original video, the estimated background, the binary mask of detected moving pixels, and the original video with detected objects on a black background.


## Task Two: Detection and Tracking of Pedestrians

This function performs object detection and tracking of pedestrians in a video using a pre-trained model. It loads the COCO class names and initializes parameters. The code processes each frame in the video, performing object detection using a deep neural network (DNN) model pre-trained on the COCO dataset. It calculates the distance to detected persons, annotates frames with bounding boxes, class names, and distances, and displays the output in a 2x2 grid. The code effectively tracks and displays information about the three closest detected persons.

## Sample Outputs:
![Output of task one](task_one_output.mp4) <br>
![Output of task two](task_two_output.mp4)


## Instructions:

To run execute following commands and press 'q' to exit the video display.

```bash
# For Task One
python movingObj.py -d carpark_f6_compressed.wm

# For Task Two
python movingObj.py -b carpark_f6_compressed.wm
