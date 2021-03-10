"""
Takes a user webcam video feed, and detects faces
**Uses fuctions from cropface.py***
"""

import cv2 
import numpy as np
from cropface import detect_face, box_faces

cap = cv2.VideoCapture(0)

while True:
	ret, frame = cap.read()
	
	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	faces = detect_face(gray)

	box_faces(frame,faces,60)

	cv2.imshow('frame',frame)
	if cv2.waitKey(1) == 27:
		break

cap.release()
cv2.destroyAllWindows() 