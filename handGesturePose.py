import cv2 

import mediapipe as mp
#https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer#get_started
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)

while webcam.isOpened():

	success, img = webcam.read()
	
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	results = mp_hands.Hands(max_num_hands=3, min_detection_confidence=0.7, min_tracking_confidence=1).process(img)
	
	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
	if results.multi_hand_landmarks:
		for hand_landmarks in results.multi_hand_landmarks:
			img=mp_drawing.draw_landmarks(img, hand_landmarks, connections=mp_hands.HANDS_CONNECTION)
	cv2.imshow('koolac', img)
	if cv2.waitKey(5) & 0xFF ==  ord('q'):
		break
	
webcam.release()
cv2.destroyAllWindows()