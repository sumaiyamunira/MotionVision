import cv2
import numpy as np
import sys



# Moving Object Detection
def perform_task_one(video_file):
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    video_capture = cv2.VideoCapture(video_file)
    frame_count = 0  # To keep track of the frame number

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        frame_count += 1
        originalVideo = frame
        frame = cv2.resize(frame, (640, 480))
        
        fg_mask = bg_subtractor.apply(frame)

        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        object_count = 0  # To count the total number of objects
        person_count = 0
        car_count = 0
        other_count = 0

        black_background = np.zeros(frame.shape, dtype=np.uint8)
        

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if 0.2 < aspect_ratio < 5:
                object_count += 1
                object_type = "Other"
                if aspect_ratio > 1:
                    object_type = "Car"
                    car_count += 1
                elif aspect_ratio < 1:
                    object_type = "Person"
                    person_count += 1
                else:
                    other_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, object_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Extract and display the detected object on a black background with the object's shape
                mask = np.zeros_like(fg_mask)
                cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
                
                detected_object = cv2.bitwise_and(originalVideo, originalVideo, mask=mask)
                black_background = cv2.add(black_background, detected_object)
                


        # Create a 2x2 grid of frames
        grid = np.zeros((480 * 2, 640 * 2, 3), dtype=np.uint8)

        # Top left: Original Video Frame
        grid[:480, :640] = originalVideo

        # Top right: Estimated Background Frame
        background_frame = bg_subtractor.getBackgroundImage()
        if background_frame is not None:
            grid[:480, 640:] = cv2.resize(background_frame, (640, 480))

        # Bottom left: Detected Moving Pixels before Filtering (in Binary Mask)
        grid[480:, :640] = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        # Bottom right: Detected Objects in original color on a black background
        grid[480:, 640:] = black_background

        # Bottom right: Detected Objects (in Original color)
        #grid[480:, 640:] = frame.copy()


        # Output the object counts to the command window
        if(object_count > 0):
             print(f"Frame {frame_count:04}: {object_count} objects ({person_count} persons, {car_count} cars, {other_count} others)")
        else:
            print(f"Frame {frame_count:04}: {object_count} objects")

        cv2.imshow('Moving Object Detection', grid)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    video_capture.release()
    cv2.destroyAllWindows()




# Detection and Tracking of Pedestrians
def perform_task_two(video_file):
    # Load the COCO class names
    with open('object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')

    # Load the DNN model
    model = cv2.dnn.readNet(
        model='frozen_inference_graph.pb',
        config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
        framework='TensorFlow'
    )

    # Capture the video
    cap = cv2.VideoCapture(video_file)

    # Get the video frames' width and height for proper saving of videos
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Create a variable to hold the original video frames
    originalVideo = None

    # Create a 2x2 grid of frames
    grid = np.zeros((480 * 2, 640 * 2, 3), dtype=np.uint8)

    # Initialize known width and focal length 
    known_width = 16  # Average shoulder width of a person in inches
    focal_length = 600  # Focal length of the camera (calibrated value)

    # Initialize lists to store the closest people
    closest_people = [{'distance': float('inf'), 'class_name': ''}] * 3

    

    # Detect objects in each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            image = cv2.resize(frame, (640, 480))
            image_height, image_width, _ = image.shape

            originalVideo = image.copy()
            second_video = image.copy()
            third_video = image.copy()
            final_video = image.copy()

            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                image=image, size=(300, 300), mean=(104, 117, 123),
                swapRB=True
            )

      
            model.setInput(blob)
            output = model.forward()


            for closest_person in closest_people:
                closest_person['distance'] = float('inf')

            # a list to store detected persons
            detected_persons = []
          
          
            # Loop over each of the detections
            for detection in output[0, 0, :, :]:
                # Extract the confidence of the detection
                confidence = detection[2]
             
                
                # Draw bounding boxes only if the detection confidence is above a certain threshold
                if confidence > 0.4:
                    # Get the class id
                    class_id = detection[1]
                    class_name = class_names[int(class_id) - 1]
                    if class_name == 'person':

                        box_x = detection[3] * image_width
                        box_y = detection[4] * image_height
                        box_width = detection[5] * image_width
                        box_height = detection[6] * image_height


                        # This condition is to remove false postive cases
                        if (box_width- box_x )> (image_width*0.6):
                            continue


                        # Calculate the distance to the detected person
                        box_width_in_pixels = box_width
                        distance = calculate_distance(known_width, focal_length, box_width_in_pixels) 

                        
                         # Store the detected person's information in the list
                        detected_persons.append({'box_x': box_x, 'box_width': box_width,'box_y': box_y, 'box_height': box_height, 'distance': distance, 'class_name': class_name})


                        # Draw a rectangle around each detected object
                        cv2.rectangle(second_video, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (0,0,255), thickness=2)

                        # Put the class name on the detected object
                        text = f"{class_name}"
                        cv2.putText(third_video, text, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                        # Draw a rectangle around each detected object
                        cv2.rectangle(third_video, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (0,255,0), thickness=2)

          
            # Sort the detected persons by distance
            detected_persons.sort(key=lambda x: x['distance'])

            # Take the three closest persons
            closest_people = detected_persons[:3]

            for idx, closest_person in enumerate(closest_people):
                class_name = closest_person['class_name']
                distance = closest_person['distance']
                if class_name == 'person':
                    # Draw a rectangle around the closest detected object
                    cv2.rectangle(final_video, (int(closest_person['box_x']), int(closest_person['box_y'])), (int(closest_person['box_width']), int(closest_person['box_height'])), (255, 0, 0), thickness=2)
                    #text = f"({distance:.2f} inches)"
                    #cv2.putText(final_video, text, (int(closest_person['box_x']), int(closest_person['box_y'] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Copy the frames into the grid
            grid[:480, :640] = originalVideo
            grid[:480, 640:] = second_video
            grid[480:, :640] = third_video
            grid[480:, 640:] = final_video

            cv2.imshow('Detection and Tracking of Pedestrians', grid)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()




def calculate_distance(known_width, focal_length, box_width_in_pixels):
    distance = (known_width * focal_length) / box_width_in_pixels
    return distance




if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: movingObj [-b/-d] video_file")
        sys.exit(1)

    option = sys.argv[1]
    video_file = sys.argv[2]

    if option == '-b':
        perform_task_one(video_file)
    elif option == '-d':
        perform_task_two(video_file)
    else:
        print("Invalid option. Please use '-b' for Task One and '-d' for Task Two.")
