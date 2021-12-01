import sys

import cv2
import pytesseract
from pytesseract import Output
import random as rng
import pandas
import numpy as np
import string
import base64
import io
from imageio import imread

from words import ALL_WORDS, two_word_mapping


def codenamify_words(word_list, contours):
    mapped_words = []
    final_contours = []
    final_words = []

    for i in range(len(word_list)):
        word = word_list[i]
        word = word.translate(str.maketrans('', '', string.punctuation))

        # both 'ice' and 'ice cream' are codenames words, so don't convert 'ice' to 'ice cream'
        # unless 'cream' directly follows it
        if word == "ice" and (i == len(word_list) - 1 or word_list[i + 1] != "cream"):
            mapped_words.append("ice")
        else:
            mapped_words.append(two_word_mapping.get(word, word))

    for i in range(len(mapped_words)):
        if(mapped_words[i] in ALL_WORDS and mapped_words[i] not in final_words):
            final_contours.append(contours[i])
            final_words.append(mapped_words[i])

    xcoords = []
    ycoords = []

    #isolate inner grid and rank contour coordinates by quantile
    for i,c in enumerate(final_contours):
        area = cv2.contourArea(c)
        if(area>40):
            (x,y),_ = cv2.minEnclosingCircle(c)
            xcoords.append(x)
            ycoords.append(y)

    yranks = pandas.qcut(ycoords,5,labels=[1, 2, 3, 4, 5])
    xranks = pandas.qcut(xcoords,5,labels=[1, 2, 3, 4, 5])

    #sort contours
    final_words, xcoords, ycoords = zip(*sorted(zip(final_words, xranks, yranks), key = lambda b:[b[2], b[1]], reverse=False))

    print(list(final_words))
    print(len(final_words))
    return list(final_words)

class gameBoard:
    def __init__(self):
        pass

    def getGameText(self, imgEncoding, use_b64_encoding=True):
        if use_b64_encoding:
            img = imread(io.BytesIO(base64.b64decode(imgEncoding)))
        else:
            img = cv2.imread(imgEncoding,cv2.IMREAD_COLOR)

        img = cv2.cvtColor(img, cv2.IMREAD_ANYCOLOR)
        img = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)


        #initialize square/rect structuring kernels 
        rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        

        #Convert image to grayscale and apply blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)

        blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, rectKernel)

        gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradX = np.absolute(gradX)
        (minVal, maxVal) = (np.min(gradX), np.max(gradX))
        gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

        #threshold image
	    
        gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
        

        # threshInv = cv2.adaptiveThreshold(gradX,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,13, 30)
        threshInv = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #cv2.imwrite('data/thresh_result.jpg', threshInv)

        p = int(img.shape[1] * 0.05)
        threshInv[:, 0:p] = 0
        threshInv[:, img.shape[1] - p:] = 0

        opening = cv2.morphologyEx(threshInv, cv2.MORPH_OPEN, sqKernel)
        #morph = cv2.morphologyEx(threshInv, cv2.MORPH_CLOSE, sqKernel)
        #cv2.imwrite('data/morph_result.jpg', threshInv)
        erode = cv2.erode(opening, None, iterations=3)

        #cv2.imwrite('data/erode_result.jpg', erode)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,9))
        dilate = cv2.dilate(erode, kernel, iterations=3)

        #cv2.imwrite('data/dilate_result.jpg', dilate)

        orig_contours, hierarcy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_poly = [None]*len(orig_contours)
        boundRect = [None]*len(orig_contours)

        for i, c in enumerate(orig_contours):
            hull = cv2.convexHull(c)
            contours_poly[i] = cv2.approxPolyDP(hull, 4, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])

        line_items_coordinates = []
        drawing = gray
        for i,c in enumerate(orig_contours):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
                (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            cv2.putText(drawing, str(i), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 2, cv2.LINE_AA)
            line_items_coordinates.append([(int(boundRect[i][0]),int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3]))])

        #cv2.imwrite('data/contour_result.jpg', drawing)

        custom_config = r'--oem 3 --psm 6'

        total_boxes = len(line_items_coordinates)
        print(total_boxes)
        game_words = []
        for sequence_number in range(total_boxes):
            c = line_items_coordinates[sequence_number]
            cropped = img[c[0][1]:c[1][1], c[0][0]:c[1][0]]  
            c_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

            ret,thresh1 = cv2.threshold(c_gray,120,255,cv2.THRESH_BINARY)
            text = str(pytesseract.image_to_string(thresh1, config=custom_config, lang='eng'))
            text = text.replace("\x0c", "")
            text = text.replace("\n", "")
            
            game_words.append(text.lower())
        print(game_words)
        return codenamify_words(game_words, orig_contours)


if __name__ == '__main__':
    game = gameBoard()
    words = game.getGameText(sys.argv[1], False)
    print(words, len(words))