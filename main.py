import cv2
from cv2 import dnn_superres
import pyautogui
import numpy as np
import imutils
# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read image
while True:
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
# Read the desired model
    path = "ESPCN_x3.pb"
    sr.readModel(path)

# Set the desired model and scale to get correct pre- and post-processing
    sr.setModel("espcn", 3)

# Upscale the image
    result = sr.upsample(image)

# Save the image
    cv2.imshow("Video Upscale", imutils.resize(result, width=800, height=600))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
