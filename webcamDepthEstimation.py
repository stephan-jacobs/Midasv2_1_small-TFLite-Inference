import cv2
import time
import numpy as np
from MidasDepthEstimation.midasDepthEstimator import midasDepthEstimator

# Initialize depth estimation model
depthEstimator = midasDepthEstimator()

# Initialize webcam
camera = cv2.VideoCapture(0)
cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL) 	

while True:
    time.sleep(2)
    # Read frame from the webcam
    ret, img = camera.read()

    # Estimate depth
    colorDepth = depthEstimator.estimateDepth(img)
    
    time.sleep(2)
    # Add the depth image over the color image:
    #combinedImg = cv2.addWeighted(img,0.7,colorDepth,0.6,0)

    # Join the input image, the estiamted depth and the combined image
    #img_out = np.hstack((img, colorDepth, combinedImg))
    img_out = np.hstack((img, colorDepth))

    cv2.imshow("Depth Image", img_out)
    #cv2.imshow("Image", img)

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()