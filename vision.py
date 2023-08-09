import sys
import cv2
import numpy as np
import utils
import calibration
from matplotlib import pyplot as plt

# Mediapipe for face detection
import mediapipe as mp

mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Open both cameras
camera1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)                    
camera2 =  cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Stereo vision setup parameters
frame_rate = 60    #Camera frame rate (maximum at 120 fps)
baseline = 9       #Distance between the cameras [cm]
f = 8              #Camera lense's focal length [mm]
fov = 56.6        #Camera field of view in the horisontal plane [degrees]

# Main program loop with face detector and depth estimation using stereo vision
with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

    while(camera1.isOpened() and camera2.isOpened()):

        ret1, frame_right = camera1.read()
        ret2, frame_left = camera2.read()

        frame_right, frame_left = utils.undistort_rectify(frame_right, frame_left)

        if not ret1 or not ret2:                    
            break

        else:
            
            # Convert the BGR image to RGB
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

            # Process the image and find faces
            results_right = face_detection.process(frame_right)
            results_left = face_detection.process(frame_left)

            # Convert the RGB image to BGR
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR)
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR)

            ################## CALCULATING DEPTH #########################################################

            center_right = 0
            center_left = 0

            if results_right.detections:
                for id, detection in enumerate(results_right.detections):
                    mp_draw.draw_detection(frame_right, detection)
                    bBox = detection.location_data.relative_bounding_box
                    h, w, c = frame_right.shape
                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                    center_point_right = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)
                    cv2.putText(frame_right, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

            if results_left.detections:
                for id, detection in enumerate(results_left.detections):
                    mp_draw.draw_detection(frame_left, detection)
                    bBox = detection.location_data.relative_bounding_box
                    h, w, c = frame_left.shape
                    boundBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                    center_point_left = (boundBox[0] + boundBox[2] / 2, boundBox[1] + boundBox[3] / 2)
                    cv2.putText(frame_left, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

            # If no ball can be caught in one camera show text "TRACKING LOST"
            if not results_right.detections or not results_left.detections:
                cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
                cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

            else:
                depth = utils.find_depth(center_point_right, center_point_left, frame_right, frame_left, baseline, fov)

                cv2.putText(frame_right, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                cv2.putText(frame_left, "Distance: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),3)
                print("Depth: ", str(round(depth,1)))
                     

            # Show the frames
            both_frames = np.concatenate((frame_left, frame_right), axis=1)
            cv2.imshow("Frames", both_frames)


            # Hit "q" to close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


# Release and destroy all windows before termination
camera1.release()
camera2.release()

cv2.destroyAllWindows()