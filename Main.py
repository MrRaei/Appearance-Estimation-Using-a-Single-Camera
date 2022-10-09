import cv2
import numpy as np
import scipy.linalg as la
import TopBottomPix_Class
import PoseEstimation_Class

from datetime import datetime
import urllib

import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths


print ("Starting ...")
from sklearn.externals import joblib

clfHeight = joblib.load("Data/Frames Data/SVM Data/5Area/clfHeight.pkl")
clfWeight = joblib.load("Data/Frames Data/SVM Data/5Area/clfWeight.pkl")
clfSex = joblib.load("Data/Frames Data/SVM Data/clfSex.pkl")

def url_to_image(url):
	# download the image, convert it to a NumPy array,
	# and then read it into OpenCV format
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)

	# return the image
	return image

url = "http://192.168.137.32:8888/shot.jpg"

topMost, bottomMost = 0, 0
Height = 0

FileDirection = "Data/Videos/3/2/"
FileName = "sepide khakzad 1_176_59"
FileFormat = ".mp4"
cap = cv2.VideoCapture(FileDirection + FileName + FileFormat)
# file = open(FileDirection + FileName + ".txt","w")
# file.write(FileName + "\n")
# file.close()

mtx, rvecs, tvecs, dist, objp = PoseEstimation_Class.PoseEstimationFirstValues()

##
KInv = la.inv(mtx)
R, _ = cv2.Rodrigues(rvecs)
RTOrginal_4_4 = np.append(np.concatenate((R, tvecs),axis=1), [0, 0, 0, 1]).reshape((4,4))

r1 = R[:,0]
r2 = R[:,1]
r3 = R[:,-1]

r1 = np.array([[r1[0]], [r1[1]], [r1[2]]])
r2 = np.array([[r2[0]], [r2[1]], [r2[2]]])
r3 = np.array([[r3[0]], [r3[1]], [r3[2]]])

R = R[:,0:2]
RT = np.concatenate((R,tvecs),axis=1)
KRT = np.dot(mtx,RT)
H = la.inv(KRT)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# make a function to find moving objects (like human)
fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=15, detectShadows=True)

print ("I'm Ready ...")

# file = open(FileDirection + FileName + ".txt","a")
FrameData = []
frameCount = 3
emptyTest = frameCount
alphaArr = []

for itr in range(0*10):
	ret, frame = cap.read()

	# if video is ended
	if ret == False:
		break

	# frame = url_to_image(url)
	
	# Background Subtraction on frame
	AUX = frame
	# AUX[0:720,0:200] = 0
	# AUX[0:720,880:1280] = 0
	fgmask = fgbg.apply(AUX, learningRate = 0.0005)
	# print (fgmask)


while True:
	# read frame from video
	ret, frame = cap.read()

	# if video is ended
	if ret == False:
		break

	# frame = url_to_image(url)
	
	# Background Subtraction on frame
	# AUX = np.zeros(frame.shape)
	# AUX[0:720,200:880] = frame[0:720,200:880]
	AUX = frame
	# AUX[0:720,0:200] = 0
	# AUX[0:720,880:1280] = 0
	fgmask = fgbg.apply(AUX, learningRate = 0.0005)
	# AUX[AUX==0] = 255
	##############################################################################
	# find Top Most Y, Bottom Most Y & X (X is same for both Top & Bottom)
	topMost, bottomMost, topBottomX, area, area1, area2, area3, area4, area5, T, width = TopBottomPix_Class.TopMostBottomMost(fgmask)
	##############################################################################
	# if find a body
	# if (bottomMost - topMost > 0 and bottomMost < 680):

	# #####################################################
	image = frame
	image = imutils.resize(image, width=min(400, image.shape[1]))

	# detect people in the image
	rects, weights = hog.detectMultiScale(image, winStride=(8, 4),padding=(4, 2), scale=1.04)
	# rects = non_max_suppression(rects, probs=None, overlapThresh=0.65)
	# print (weights)
	# print (rects)
	# #####################################################

	if (topMost > 10 and bottomMost < 710 and len(weights) == 1):
		Height, distansFromCamera = PoseEstimation_Class.PoseEstimation(H, KInv, RTOrginal_4_4, r1, r2, r3, tvecs, topMost, bottomMost, topBottomX)
		# print (distansFromCamera)
		width = float(bottomMost - topMost)/float(width)
		alphaArr.append([width])

		# frame = PoseEstimation_Class.Cube(frame, Height, mtx, dist, objp)
		# print (Height)
		if Height > 50:

			# # Top
			cv2.line(frame,(topBottomX-((bottomMost-topMost)//4),topMost),(topBottomX+((bottomMost-topMost)//4),topMost),(0,0,255),3)
			# # Bottom
			cv2.line(frame,(topBottomX-((bottomMost-topMost)//4),bottomMost),(topBottomX+((bottomMost-topMost)//4),bottomMost),(0,0,255),3)
			# # Left
			cv2.line(frame,(topBottomX-((bottomMost-topMost)//4),topMost),(topBottomX-((bottomMost-topMost)//4),bottomMost),(0,0,255),3)
			# # Right
			cv2.line(frame,(topBottomX+((bottomMost-topMost)//4),topMost),(topBottomX+((bottomMost-topMost)//4),bottomMost),(0,0,255),3)

			# Height
			# cv2.putText(frame,"%d cm"%Height,(topBottomX-150,topMost),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

			# if emptyTest == 0:
			# 	FrameData = []
			# 	emptyTest = frameCount
			# emptyTest -= 1
			FrameData.append([float(Height), float((area1)*(distansFromCamera**2)), float((area2)*(distansFromCamera**2)), 
				float((area3)*(distansFromCamera**2)), float((area4)*(distansFromCamera**2)), float((area5)*(distansFromCamera**2))])

			# Height
			hErr = 1.02
			if width < 3:
				hErr = 1.025
			if width < 2.9:
				hErr = 1.03
			if width < 2.8:
				hErr = 1.005
			hErr = 1.03
			# cv2.putText(frame,"%d cm"%round(np.mean(clfHeight.predict([[x[0]] for x in FrameData])) * hErr),(topBottomX-((bottomMost-topMost)/4),topMost),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
			cv2.putText(frame,"Height: %d cm"%round(np.median(clfHeight.predict([[x[0]] for x in FrameData])) * hErr),(10,40),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
			# Weight
			wErr = 0.94
			# if width > 132000:
			# 	wErr = 0.99
			# print (width)
			if width < 3:
				wErr = 1.1
			if width < 2.9:
				wErr = 1.15
			if width < 2.8:
				wErr = 1.4
			wErr = 0.94
			cv2.putText(frame,"Weight: %d kg"%round((np.mean(clfWeight.predict(FrameData))) * wErr),(10,80),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)

			# cv2.putText(frame,"pc: %d (cm)"%Height,(10,120),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

			crop_img = frame[topMost:bottomMost, topBottomX-((bottomMost-topMost)//4):topBottomX+((bottomMost-topMost)//4)].copy()
			crop_img = cv2.resize(crop_img, (64, 128))
			imgFeature = hog.compute(crop_img).squeeze()
			gender = clfSex.predict([imgFeature])
			if gender < 0.5:
				cv2.putText(frame,"Gender: Male",(10,120),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
			else:
				if gender > 0.5:
					cv2.putText(frame,"Gender: Female",(10,120),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
				else:
					cv2.putText(frame,"Gender: Unknow",(10,120),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
			# frame[T==1,2] = 255


	else:
		Height = 0
		# cv2.putText(frame,"Height: %d (cm)"%Height,(10,40),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
		# cv2.putText(frame,"Weight: %d (kg)"%0,(10,80),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,255),2)
		# cv2.putText(frame,"Gender: %s"%"Male",(10,120),cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)

	cv2.imshow('Human',AUX)
	k = cv2.waitKey(1) & 0xff
	# break Key
	if k == ord('q'):
		break
	if k == ord('Q'):
		break

# print (np.max(alphaArr))
# print (np.min(alphaArr))
# print (np.median(alphaArr))
# print (np.mean(alphaArr))
cap.release()
cv2.destroyAllWindows()

print('End')