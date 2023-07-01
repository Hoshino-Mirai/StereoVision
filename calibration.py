import cv2
import numpy as np

camera1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

def image_captures():
    while True:
        ret1, frame1 = camera1.read()
        ret2, frame2 = camera2.read()
        
        if (ret1 and ret2):
            both_frames = np.concatenate((frame1, frame2), axis=1)
            cv2.imshow('Frames', both_frames)
        else:
            print('Error reading cameras')
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img)
            cv2.imwrite('images/stereoright/imageR' + str(num) + '.png', img2)
            

    camera1.release()
    camera2.release()

    cv2.destroyAllWindows()
    

