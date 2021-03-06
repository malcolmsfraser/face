"""
Takes a user screen-capture video feed, and detects faces
**Uses fuctions from cropface.py***
"""

import cv2 
import numpy as np
from PIL import ImageGrab
from cropface import detect_face, box_faces


while True:
	cap = ImageGrab.grab()
	cap_np = np.array(cap)
	
	frame = cv2.cvtColor(cap_np,cv2.COLOR_BGR2RGB)
	gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
	faces = detect_face(gray)

	box_faces(frame,faces)

	resize = cv2.resize(frame, (854,480))
	cv2.imshow('frame',resize)
	if cv2.waitKey(1) == 27:
		break

cv2.destroyAllWindows() 