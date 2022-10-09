import numpy as np
import cv2
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
kernelc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
# kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,5))
# kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))

def TopMostBottomMost(fgmask):
	# A Threshold for length of human full body
	# humanBodyLenThreshold = 500
	# topMost, bottomMost = 0, 0

	######################################################################
	# Remove Shadows (Shadows are 127 in fgmask array)
	# cv2.imshow("Background Subtraction",fgmask)
	fgmask[fgmask < 255] = 0
	
	T = fgmask.astype(np.float32) / 255	# convert to floating point

	# Remove Noise
	## threshold
	# thresh = .13
	# re, T = cv2.threshold(fgmask,thresh,1,cv2.THRESH_BINARY)
	# cv2.imshow("",fgmask)

	T = fgmask/255

	T = np.array(T, dtype=np.uint8)

	nC,C,stats, centroids=cv2.connectedComponentsWithStats(T);

	L = np.zeros(T.shape)

	for i in range(1,nC):
		if stats[i,4] > 10000:
			L[C==i] = 1
			# print (stats[i,4])

	T = L
	# Connect whole Parts of a Object (body)
	## opening
	# kernel = np.ones((5,3),np.uint8)
	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,3))
	# T = cv2.morphologyEx(T, cv2.MORPH_OPEN, kernel)
	
	## closing
	# kernel = np.ones((30,30),np.uint8)
	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	# T = cv2.morphologyEx(T, cv2.MORPH_CLOSE, kernelc)
	T = cv2.morphologyEx(T, cv2.MORPH_CLOSE, kernelc)


	# cv2.imshow('Computer',T)
	######
	area = np.count_nonzero(T)
	area1 = 0
	area2 = 0
	area3 = 0
	area4 = 0
	area5 = 0

	width = 0

	h = np.nonzero(T.max(axis=1))[0] # All non zero Y
	# print (h)
	# if you find an object (body)
	if h != [] and h[0] > 10 and h[-1] < 710: #len(h) > humanBodyLenThreshold:
				
		# print (np.count_nonzero(T))
		topMost = h[0] # Top Y
		topBand = np.nonzero(T[topMost,:]) # an array with Top Y and all X
		bandMedian = int(np.median(topBand)) # Median X in topBand array
		topBottomX = bandMedian # Top & Bottom X
		# make a length band
		# newT = T[ : , bandMedian-10 : bandMedian+10] # a area with Limeted X and all Y

		# print (newT)
		# if newT != []:
		# 	newH = np.nonzero(newT.max(axis=1))[0] # All non zero Y in area
		# 	bottomMost = newH[-1] # Bottom Y
		# 	bottomMost = h[-1]

		bottomMost = h[-1]

		partArea = (bottomMost - topMost)//5

		# Area
		topArea = topMost
		botArea = topArea + partArea
		area1 = np.count_nonzero(T[topArea : botArea , : ])

		topArea = botArea
		botArea = botArea + partArea
		area2 = np.count_nonzero(T[topArea : botArea , : ])

		alpha = (np.nonzero((T[topArea : botArea , : ]).max(axis=0))[0])[-1] - (np.nonzero((T[topArea : botArea , : ]).max(axis=0))[0])[0]
		width = width + alpha
		# if width < alpha:
		# 	width = alpha

		topArea = botArea
		botArea = botArea + partArea
		area3 = np.count_nonzero(T[topArea : botArea , : ])

		alpha = (np.nonzero((T[topArea : botArea , : ]).max(axis=0))[0])[-1] - (np.nonzero((T[topArea : botArea , : ]).max(axis=0))[0])[0]
		width = width + alpha
		# if width < alpha:
		# 	width = alpha

		topArea = botArea
		botArea = botArea + partArea
		area4 = np.count_nonzero(T[topArea : botArea , : ])

		alpha = (np.nonzero((T[topArea : botArea , : ]).max(axis=0))[0])[-1] - (np.nonzero((T[topArea : botArea , : ]).max(axis=0))[0])[0]
		width = width + alpha
		# if width < alpha:
		# 	width = alpha

		topArea = botArea
		botArea = bottomMost
		area5 = np.count_nonzero(T[topArea : botArea , : ])

		width = width/3

	else:
		topMost, bottomMost, topBottomX = 0, 0, 0
	######

	# fgmask = T
	######################################################################
	
	return topMost, bottomMost, topBottomX, area, area1, area2, area3, area4, area5, T, width
	