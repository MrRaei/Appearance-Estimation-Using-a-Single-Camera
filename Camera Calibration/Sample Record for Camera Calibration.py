from datetime import datetime
import numpy as np
import urllib
import cv2

def url_to_image(url):
	# download the image, convert it to a NumPy array,
	# and then read it into OpenCV format
	resp = urllib.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)

	# return the image
	return image



url = "http://192.168.43.158:8888/shot.jpg"

while True:

	# get image from IP Camera
	image = url_to_image(url)

	# display image
	cv2.imshow("Image", image)
	if cv2.waitKey(1000) & 0xFF == ord('q'):
		break

	# save image to file, if pattern found
	ret, corners = cv2.findChessboardCorners(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), (7,6), None)

	if ret == True:
		filename = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss%f') + '.jpg'
		cv2.imwrite("Sample Calibration Images/" + filename, image)


cv2.destroyAllWindows()
