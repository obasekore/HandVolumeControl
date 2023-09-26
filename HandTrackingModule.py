import cv2
import mediapipe as mp
import time
import numpy as np

#https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer#get_started



class handDetector():

	def __init__(self, mode=False, maxHand=2, detectionCon=0.5, trackCon=0.5):
	
		self.mode = mode
		self.maxHands = maxHand
		self.detectionCon = detectionCon
		self.trackCon = trackCon

		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(self.mode, max_num_hands = self.maxHands, min_detection_confidence = self.detectionCon, min_tracking_confidence = self.trackCon)
		self.mpDraw = mp.solutions.drawing_utils


	def findHands(self, img, draw=True):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		self.results = self.hands.process(imgRGB)

		if self.results.multi_hand_landmarks:
			for hand_landmarks in self.results.multi_hand_landmarks:
				if draw:
					self.mpDraw.draw_landmarks(img, hand_landmarks, connections=self.mpHands.HAND_CONNECTIONS)
		return img

		
	def findPosition(self, img, handNo=0, draw=True):
		lmList = []
		if self.results.multi_hand_landmarks:
			myHand = self.results.multi_hand_landmarks[handNo]
			for id, lm in enumerate(myHand.landmark):
				h, w, c = img.shape

				cx, cy = int(lm.x * w), int(lm.y * h)

				lmList.append([id, cx, cy])

				if draw:
					cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
		return lmList
		