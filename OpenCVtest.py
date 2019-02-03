import numpy as np
import cv2 as cv
import glob

def nothing(x):
    pass


img_RWBB_max_val = []

# значения поумолчанию
red_frame = [(0, 215, 140), (40, 255, 255)]
white_frame = [(80, 240, 115), (255, 255, 255)]
blue_frame = [(0, 0, 120), (180, 60, 255)]
black_frame = [(0, 0, 0), (30, 110, 70)]

accuracy_black = 0.5
accuracy_white = 0.5
accuracy_blue = 0.5
accuracy_red = 0.5
accuracy_all = 0.2

min_set = (0, 0, 0)
max_set = (1, 1, 1)
erode_weight = [0, 0, 0, 0]
dilate_weight = [0, 0, 0, 0]
blur_weight = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

print('\t\t\tНачало настройки\nr-для отделения красного\nw-белого\nc-синего\nb-черного\n0-Конец\n')

setting_true = True

cv.namedWindow('tuning', cv.WINDOW_AUTOSIZE)
cv.namedWindow('frame', cv.WINDOW_AUTOSIZE)
cv.namedWindow('hsv', cv.WINDOW_AUTOSIZE)

cv.createTrackbar('dilate', 'tuning', 0, 25, nothing)
cv.createTrackbar('erode', 'tuning', 0, 25, nothing)
cv.createTrackbar('blur', 'frame', 0, 25, nothing)

cv.createTrackbar('minB', 'hsv', 0, 255, nothing)
cv.createTrackbar('minG', 'hsv', 0, 255, nothing)
cv.createTrackbar('minR', 'hsv', 0, 255, nothing)
cv.createTrackbar('maxB', 'hsv', 0, 255, nothing)
cv.createTrackbar('maxG', 'hsv', 0, 255, nothing)
cv.createTrackbar('maxR', 'hsv', 0, 255, nothing)

font = cv.FONT_HERSHEY_SIMPLEX
cam = cv.VideoCapture(0)

files = glob.glob("./z?*Img/?.jpg")
def count_MaxVal(img1, val=0):    # определения кол-во пикселей нужного цвета в эталонах
    for i in range(64):
        for j in range(64):
            if img1[i][j] == 255:
                val += 1
    return val

# суммирование схождение цвета с эталоном
def count_white(img1, img2, white_count=0, val=0):
    for i in range(64):
        for j in range(64):
            if white_count == 0:
                if img1[i][j] == 255:
                    if img1[i][j] == img2[i][j]:
                        val += 1
            else:
                if img1[i][j] == 255:
                        val += 1
    return val


# распредеение по цветам
for myFile in files:
    image = cv.imread(myFile)
    image = cv.inRange(image, (50, 50, 50), (255, 255, 255))
    img_RWBB_max_val.append(count_MaxVal(image))
print(img_RWBB_max_val)

def test_ver(Img, indexfile, min, max, img_max_val, accuracy, blur, erode, dilate, counter, divisor):

    ImgCheck = cv.blur(Img, blur)  # размытие для уменешения шумов
    ImgCheck = cv.inRange(ImgCheck, min, max)
    ImgCheck = cv.erode(ImgCheck, None, iterations=erode)  # избавление от одиноких пикселей
    ImgCheck = cv.dilate(ImgCheck, None, iterations=dilate)

    ImgProto = cv.inRange(cv.imread(indexfile), (50, 50, 50), (255, 255, 255))
    if img_max_val > 0:
        val = count_white(ImgCheck, ImgProto)
        brAns = val / img_max_val
        if brAns >= accuracy:
            print(brAns, ' - ', accuracy)
            return brAns
        else:
            return 0
    else:
        divisor[counter % 4] -= 1
        return 0

def recognition(Img, counter=0):
    ImgCheck = Img.copy()
    divisor = [4, 4, 4, 4]
    ansCheck = []
    full_answer = [0, 0, 0, 0]
    for index in files:
        if counter < 4:       # красный
            min = red_frame[0]
            max = red_frame[1]
            accuracy = accuracy_red
            blur = blur_weight[0]
            erode = erode_weight[0]
            dilate = dilate_weight[0]
            print('RED')
        elif counter < 8:     # белый
            min = white_frame[0]
            max = white_frame[1]
            accuracy = accuracy_white
            blur = blur_weight[1]
            erode = erode_weight[1]
            dilate = dilate_weight[1]
            print('WHITE')
        elif counter < 12:    # синий
            min = blue_frame[0]
            max = blue_frame[1]
            accuracy = accuracy_blue
            blur = blur_weight[2]
            erode = erode_weight[2]
            dilate = dilate_weight[2]
            print('BLUE')
        else:               # черный
            min = black_frame[0]
            max = black_frame[1]
            accuracy = accuracy_black
            blur = blur_weight[3]
            erode = erode_weight[3]
            dilate = dilate_weight[3]
            print('BLACK')
        ansCheck.append(test_ver(ImgCheck, index, min, max, img_RWBB_max_val[counter], accuracy, blur, erode, dilate, counter, divisor))
        counter += 1

    for index in range(16):
        if index / 4 < 1:
            full_answer[index] += ansCheck[index] / divisor[index]
        else:
            if index % 4 == 0:
                full_answer[index % 4] += ansCheck[index-4] / divisor[index % 4]
        # print('recognition -- ', full_answer[index % 4], ' += ', ansCheck[index-4], ' / ', divisor[index % 4])

    ans = 0
    for index in range(1, 4):  # нахождение наибольшего
        if full_answer[index] > full_answer[ans]:
            ans = index

    if full_answer[ans] > accuracy_all:
        # print("answer - ", ans, " = ", full_answer[ans]*100, "%")
        return ans
    elif test_ver(ImgCheck, files[2], red_frame[0], red_frame[1], img_RWBB_max_val[0], accuracy_red, blur_weight[0], erode_weight[0], dilate_weight[0], counter, divisor) > accuracy_all:
        return 4
    else:
        return 8


while True:
    ret, frame = cam.read()
    frameCopy = frame.copy()

    if setting_true:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # изменение цветовой схемы с rgb на hsv

        minB = cv.getTrackbarPos('minB', 'hsv')
        minG = cv.getTrackbarPos('minG', 'hsv')
        minR = cv.getTrackbarPos('minR', 'hsv')

        maxB = cv.getTrackbarPos('maxB', 'hsv')
        maxG = cv.getTrackbarPos('maxG', 'hsv')
        maxR = cv.getTrackbarPos('maxR', 'hsv')

        er_w = cv.getTrackbarPos('erode', 'tuning')
        dil_w = cv.getTrackbarPos('dilate', 'tuning')
        blur_w = cv.getTrackbarPos('blur', 'frame')

        min_set = (minB, minG, minR)
        max_set = (maxB, maxG, maxR)
        blur_wm = (blur_w+1, blur_w+1)

        mask_setting = cv.blur(hsv, blur_wm)
        mask_setting = cv.inRange(mask_setting, min_set, max_set)  # отделение цвета в диапозоне
        mask_setting = cv.erode(mask_setting, None, iterations=er_w)  # избавление от одиноких пикселей
        mask_setting = cv.dilate(mask_setting, None, iterations=dil_w)  # увеличение оставшихся пикселей
        cv.imshow('frame', frame)
        cv.imshow('hsv', hsv)
        cv.imshow('tuning', mask_setting)

        if cv.waitKey(1) == ord("r"):
            erode_weight[0] = er_w
            dilate_weight[0] = dil_w
            blur_weight[0] = blur_wm
            red_frame[0] = min_set
            red_frame[1] = max_set
            print('Красный настроен')
        elif cv.waitKey(1) == ord("w"):
            erode_weight[1] = er_w
            blur_weight[1] = blur_wm
            dilate_weight[1] = dil_w
            white_frame[0] = min_set
            white_frame[1] = max_set
            print('Белый настроен')
        elif cv.waitKey(1) == ord("c"):
            erode_weight[2] = er_w
            blur_weight[2] = blur_wm
            dilate_weight[2] = dil_w
            blue_frame[0] = min_set
            blue_frame[1] = max_set
            print('Синий настроен')
        elif cv.waitKey(1) == ord("b"):
            erode_weight[3] = er_w
            dilate_weight[3] = dil_w
            blur_weight[3] = blur_wm
            black_frame[0] = min_set
            black_frame[1] = max_set
            print('Черный настроен')
        elif cv.waitKey(1) == ord("0"):
            setting_true = False
            cv.destroyAllWindows()
            cv.namedWindow('frame', cv.WINDOW_NORMAL)
            cv.createTrackbar('accuracy R', 'frame', 50, 100, nothing)
            cv.createTrackbar('accuracy W', 'frame', 50, 100, nothing)
            cv.createTrackbar('accuracy C', 'frame', 50, 100, nothing)
            cv.createTrackbar('accuracy B', 'frame', 50, 100, nothing)
            cv.createTrackbar('accuracy__', 'frame', 20, 100, nothing)
            print('\t\t\tКонец настройки')
        elif cv.waitKey(1) == ord("q"):
            break

    else:
        accuracy_white = cv.getTrackbarPos('accuracy W', 'frame')/100
        accuracy_black = cv.getTrackbarPos('accuracy B', 'frame')/100
        accuracy_blue = cv.getTrackbarPos('accuracy C', 'frame')/100
        accuracy_red = cv.getTrackbarPos('accuracy R', 'frame')/100
        accuracy_all = cv.getTrackbarPos('accuracy__', 'frame')/100

        mask = cv.blur(frame, blur_weight[0])                                       # размытие для уменешения шумов
        mask = cv.cvtColor(mask, cv.COLOR_BGR2HSV)                                  # перевод в hsv
        mask = cv.inRange(mask, red_frame[0], red_frame[1])                         # отделение цвета в диапозоне
        mask = cv.erode(mask, None, iterations=erode_weight[0])                     # избавление от одиноких пикселей
        mask = cv.dilate(mask, None, iterations=dilate_weight[0])                   # увеличение оставшихся пикселей
        cv.imshow("MASK", mask)

        contour = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)         # выделение контуров
        contour = contour[1]
        countInd = 0
        if contour:
            contour = sorted(contour, key=cv.contourArea, reverse=True)             # сортировка контуров по убыванию
            # cv.drawContours(frame, contour, -1, (255, 0, 255), 2)
            countInd = len(contour)
            for index in range(countInd):
                approx = cv.approxPolyDP(contour[index], 0.02*cv.arcLength(contour[index], True), True)     # находим среди контуров что-то похожее на круг

                if 8 >= len(approx) <= 16:
                    (x, y, w, h) = cv.boundingRect(contour[index])                  # координаты прямоугольника

                    roImg = frameCopy[y:y + h, x:x + w]                             # вырезаем прямоугольник
                    showing = roImg.copy()
                    roImg = cv.cvtColor(roImg, cv.COLOR_BGR2HSV)

                    roImg = cv.resize(roImg, (64, 64))                              # изменение разрешение полученой картинки
                    answer = recognition(roImg)
                    if answer == 4:
                        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 4)
                    elif answer == 3:
                        cv.putText(frame, 'speed limit ', (x, y), font, 1, (0, 0, 0), 2)
                        cv.imshow("speed limit", showing)
                        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 4)
                    elif answer == 2:
                        cv.putText(frame, 'Movement Prohibition', (x, y), font, 1, (255, 255, 255), 2)
                        cv.imshow("Movement Prohibition", showing)
                        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 4)
                    elif answer == 1:
                        cv.putText(frame, 'STOP PROHIBITED', (x, y), font, 1, (0, 0, 255), 2)
                        cv.imshow("do not enter", showing)
                        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 4)
                    elif answer == 0:
                        cv.putText(frame, 'do not enter', (x, y), font, 1, (255, 0, 0), 2)
                        cv.imshow("STOP PROHIBITED", showing)
                        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)
                    else:
                        print('nothing')
                    cv.imshow('frame', frame)
                    break
    if cv.waitKey(1) == ord("q"):  # кнопка закрытия
        break

cv.waitKey(0)
cam.release()
cv.destroyAllWindows()
