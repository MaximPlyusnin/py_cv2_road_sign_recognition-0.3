import cv2 as cv
import numpy as np
import glob

# подготовка эталонов для анализа
img = []
Rimg = []
Wimg = []
BUimg = []
BAimg = []

red = [(0, 130, 40), (40, 255, 255)]
white = [(0, 0, 120), (180, 60, 255)]
blue = [(80, 255, 115), (255, 255, 255)]
black = [(0, 0, 0), (30, 110, 70)]

erode_w = 4
dilate_w = 6

file = glob.glob("D:/Project/py/OpenCvCource/WSourceImg/*.jpg")
file_red = glob.glob("D:/Project/py/OpenCvCource/z1redImg/[0-9].jpg")
file_white = glob.glob("D:/Project/py/OpenCvCource/z1whiteImg/[0-9].jpg")
file_blue = glob.glob("D:/Project/py/OpenCvCource/z1blueImg/[0-9].jpg")
file_black = glob.glob("D:/Project/py/OpenCvCource/z1blackImg/[0-9].jpg")

counter = 0
file_len = len(file)
for index in inRange(file)
	img[counter] = cv.imread(file)
	img[counter] = cv.cvtColor(img[counter], cv.COLOR_BGR2HSV)

	Rimg[counter] = cv.inRange(img[counter], red[0], red[1])
	Rimg[counter] = cv.erode(Rimg[counter], None, iterations=erode_w)
	Rimg[counter] = cv.dilate(Rimg[counter], None, iterations=dilate_w)
	Rimg[counter] = cv.resize(Rimg[counter], (64, 64))

	Wimg[counter] = cv.inRange(img[counter], white[0], white[1])
	Wimg[counter] = cv.erode(Wimg[counter], None, iterations=erode_w)
	Wimg[counter] = cv.dilate(Wimg[counter], None, iterations=dilate_w)
	Wimg[counter] = cv.resize(Wimg[counter], (64, 64))

	BUimg[counter] = cv.inRange(img[counter], blue[0], blue[1])
	BUimg[counter] = cv.erode(BUimg[counter], None, iterations=erode_w)
	BUimg[counter] = cv.dilate(BUimg[counter], None, iterations=dilate_w)
	BUimg[counter] = cv.resize(BUimg[counter], (64, 64))
	
	BAimg[counter] = cv.inRange(img[counter], black[0], black[1])
	BAimg[counter] = cv.erode(BAimg[counter], None, iterations=erode_w)
	BAimg[counter] = cv.dilate(BAimg[counter], None, iterations=dilate_w)
	BAimg[counter] = cv.resize(BAimg[counter], (64, 64))

	cv.imwrite(file_red[counter], Rimg[counter])
	cv.imwrite(file_white[counter], Wimg[counter])
	cv.imwrite(file_blue[counter], BUimg[counter])
	cv.imwrite(file_black[counter], BAimg[counter])
	counter += 1