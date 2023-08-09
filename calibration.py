import cv2
import os
import numpy as np
import glob

camera1 = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Right Camera
camera2 = cv2.VideoCapture(1, cv2.CAP_DSHOW) # Left Camera

demo = True

def camera_identification():
    while True:
        _, frame1 = camera1.read()
        cv2.imshow("camera1", frame1)
        k = cv2.waitKey(5)
        if k == ord('q'):
            break
    camera1.release()
    cv2.destroyAllWindows()
    
    while True:
        _, frame2 = camera2.read()
        cv2.imshow("camera2", frame2)
        k = cv2.waitKey(5)
        if k == ord('q'):
            break
    camera2.release()
    cv2.destroyAllWindows()

def image_captures():
    num = 0
    while True:
        ret1, frame1 = camera1.read()
        ret2, frame2 = camera2.read()
        
        if (ret1 and ret2):
            both_frames = np.concatenate((frame1, frame2), axis=1)
            cv2.imshow('Capturing Stage', both_frames)
        else:
            print('Error reading cameras')
            break
        
        k = cv2.waitKey(5)
        if k == ord('q'):
            break
        elif k  == ord('s'):
            # get the current directory
            current_dir = os.getcwd()

            # build the paths for the left and right images
            os.makedirs('images/left', exist_ok=True)
            os.makedirs('images/right', exist_ok=True)
            leftPath = os.path.join(current_dir, 'images', 'left', 'imageL' + str(num) + '.png')
            rightPath = os.path.join(current_dir, 'images', 'right', 'imageR' + str(num) + '.png')

            # save the images
            cv2.imwrite(leftPath, frame1)
            cv2.imwrite(rightPath, frame2)

            print ('Images saved')
            num += 1
            

    camera1.release()
    camera2.release()

    cv2.destroyAllWindows()

def calibration():
    chessboardSize = (8,5)
    frameSize = (640,480)

    objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

    wrldpoints = [] 
    imgpointsL = [] 
    imgpointsR = [] 

    current_dir = os.getcwd()
    leftPath = os.path.join(current_dir, 'images', 'left', '*.png')
    rightPath = os.path.join(current_dir, 'images', 'right', '*.png')
    imagesLeft = glob.glob(leftPath)
    imagesRight = glob.glob(rightPath)

    ctr = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0005)
    
    num = 0
    for imgLeft, imgRight in zip(imagesLeft, imagesRight):

        imgL = cv2.imread(imgLeft)
        imgR = cv2.imread(imgRight)
        grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        retL, cornersL = cv2.findChessboardCorners(grayL, chessboardSize, None)
        retR, cornersR = cv2.findChessboardCorners(grayR, chessboardSize, None)
        
        # If found, add object points, image points (after refining them)
        if (retL and retR):
            
            wrldpoints.append(objp)

            cornersL = cv2.cornerSubPix(grayL, cornersL, (15, 15), (-1, -1), ctr)
            imgpointsL.append(cornersL)

            cornersR = cv2.cornerSubPix(grayR, cornersR, (15, 15), (-1, -1), ctr)
            imgpointsR.append(cornersR)

            if(demo):
                # Draw and display the corners
                cv2.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
                cv2.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
                both_frames = np.concatenate((imgL, imgR), axis=1)
                cv2.imshow("Frames", both_frames)
                
                os.makedirs('chessBoards/left', exist_ok=True)
                os.makedirs('chessBoards/right', exist_ok=True)
                leftPath = os.path.join(current_dir, 'chessBoards', 'left', 'chessBoardsL' + str(num) + '.png')
                rightPath = os.path.join(current_dir, 'chessBoards', 'right', 'chessBoardsR' + str(num) + '.png')
                cv2.imwrite(leftPath, imgL)
                cv2.imwrite(rightPath, imgR)
                num += 1
                
                cv2.waitKey(1000)
                
    cv2.destroyAllWindows()

    ########### Calibration ###########

    _, cameraMatrixL, distortionL, _, _ = cv2.calibrateCamera(wrldpoints, imgpointsL, frameSize, None, None)
    heightL, widthL, _ = imgL.shape
    newCameraMatrixL, _ = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distortionL, (widthL, heightL), 1, (widthL, heightL))

    _, cameraMatrixR, distortionR, _, _ = cv2.calibrateCamera(wrldpoints, imgpointsR, frameSize, None, None)
    heightR, widthR, _ = imgR.shape
    newCameraMatrixR, _ = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distortionR, (widthR, heightR), 1, (widthR, heightR))

    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC

    _, newCameraMatrixL, distortionL, newCameraMatrixR, distortionR, rot, trans, _, _ = cv2.stereoCalibrate(wrldpoints, imgpointsL, imgpointsR, newCameraMatrixL, distortionL, newCameraMatrixR, distortionR, grayL.shape[::-1], ctr, flags)

    ########### Rectification ###########

    rectifyScale = 1
    rectL, rectR, projMatrixL, projMatrixR, _, _, _= cv2.stereoRectify(newCameraMatrixL, distortionL, newCameraMatrixR, distortionR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

    stereoMapL = cv2.initUndistortRectifyMap(newCameraMatrixL, distortionL, rectL, projMatrixL, grayL.shape[::-1], cv2.CV_16SC2)
    stereoMapR = cv2.initUndistortRectifyMap(newCameraMatrixR, distortionR, rectR, projMatrixR, grayR.shape[::-1], cv2.CV_16SC2)

    print ("Saving parameters!")
    cv_file = cv2.FileStorage('stereoMap.xml', cv2.FILE_STORAGE_WRITE)

    cv_file.write('stereoMapL_x',stereoMapL[0])
    cv_file.write('stereoMapL_y',stereoMapL[1])
    cv_file.write('stereoMapR_x',stereoMapR[0])
    cv_file.write('stereoMapR_y',stereoMapR[1])

    cv_file.release()
    print ("Saving completed! ")
    
# camera_identification()

# image_captures()
# calibration()

