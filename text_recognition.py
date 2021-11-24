import sys

import cv2
import pytesseract
from pytesseract import Output
import random as rng
import pandas
import numpy as np
import base64
import io
from imageio import imread

from words import ALL_WORDS, two_word_mapping


def codenamify_words(word_list):
    mapped_words = []

    for i in range(len(word_list)):
        word = word_list[i]

        # both 'ice' and 'ice cream' are codenames words, so don't convert 'ice' to 'ice cream'
        # unless 'cream' directly follows it
        if word == "ice" and (i == len(word_list) - 1 or word_list[i + 1] != "cream"):
            mapped_words.append("ice")
        else:
            mapped_words.append(two_word_mapping.get(word, word))

    return sorted(list(set(ALL_WORDS) & set(mapped_words)), key=mapped_words.index)


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

        #Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (9,9), 0)
        threshInv = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,30)

        # Dilate to combine adjacent text contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
        dilate = cv2.dilate(threshInv, kernel, iterations=4)

        orig_contours, hierarcy = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours = []
        xcoords = []
        ycoords = []

        #isolate inner grid and rank contour coordinates by quantile
        for i,c in enumerate(orig_contours):
            area = cv2.contourArea(c)
            if(area>10):
                contours.append(orig_contours[i])
                (x,y),_ = cv2.minEnclosingCircle(c)
                xcoords.append(int(x))
                ycoords.append(int(y))

        yranks = pandas.qcut(ycoords,5,labels=[1, 2, 3, 4, 5])
        xranks = pandas.qcut(xcoords,5,labels=[1, 2, 3, 4, 5])


        #sort contours
        contours, xcoords, ycoords = zip(*sorted(zip(contours, xranks, yranks), key = lambda b:[b[2], b[1]], reverse=False))


        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)

        for i, c in enumerate(contours):
            hull = cv2.convexHull(c)
            contours_poly[i] = cv2.approxPolyDP(hull, 4, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])

        line_items_coordinates = []
        drawing = np.zeros((threshInv.shape[0], threshInv.shape[1], 3), dtype=np.uint8)
        for i,c in enumerate(contours):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
                (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
            cv2.putText(drawing, str(i), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 2, cv2.LINE_AA)
            line_items_coordinates.append([(int(boundRect[i][0]),int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3]))])

        # cv2.imshow("Contour Regions", drawing)

        # cv2.waitKey(0)

        # cv2.destroyAllWindows()
        custom_config = r'--oem 3 --psm 6'

        total_boxes = len(line_items_coordinates)
        print(total_boxes)
        game_words = []
        for sequence_number in range(total_boxes):
            c = line_items_coordinates[sequence_number]
            cropped = img[c[0][1]:c[1][1], c[0][0]:c[1][0]]  
            ret,thresh1 = cv2.threshold(cropped,120,255,cv2.THRESH_BINARY)
            text = str(pytesseract.image_to_string(thresh1, config=custom_config, lang='eng'))
            text = text.replace("\n\x0c", "")
            game_words.append(text.lower())

        print(codenamify_words(game_words))
        return codenamify_words(game_words)


# if __name__ == '__main__':
#     game = gameBoard()
#     words = game.getGameText(sys.argv[1], False)
#     print(words, len(words))

#     word_list = ["loch", "ness", "ice", "cream", "new", "york", "ice", "scuba", "diver"]
#     print(codenamify_words(word_list))
