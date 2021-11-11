import sys

import cv2
import pytesseract
from pytesseract import Output
import random as rng
import numpy as np
import pandas
import base64
import io
from imageio import imread

from words import WORDS

class gameBoard:
    def __init__(self):
        pass

    def getGameText(self, imgEncoding):
        #img= cv2.imread(imgEncoding,cv2.IMREAD_COLOR)
        img = imread(io.BytesIO(base64.b64decode(imgEncoding)))
        img = cv2.cvtColor(img, cv2.IMREAD_ANYCOLOR)
        img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)

        #Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply dilation and erosion to remove some noise
        kernel = np.ones((1, 1), np.uint8)    
        dilate = cv2.dilate(gray, kernel, iterations=1)    
        erode = cv2.erode(dilate, kernel, iterations=1)
        sharp = cv2.bilateralFilter(erode, 13, 75, 75)

        #Threshold image (convert to binary). All pixels with value greater than T (70) are set to 255 (white). Otherwise, set to black (0).
        threshInv = cv2.adaptiveThreshold(sharp, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 19, 17)



        # cv2.imshow("Threshold Binary Inverse", threshInv)

        # cv2.waitKey(0)

        # cv2.destroyAllWindows()


        #psm 6 assumes a single block of text
        #oem 3 uses default engine mode
        custom_config = r'--oem 3 --psm 6'

        result = pytesseract.image_to_data(threshInv, output_type=Output.DICT,  config=custom_config, lang='eng')
        game_words = []


        total_boxes = len(result['text'])
        for sequence_number in range(total_boxes):
            #Check confidence score
            if int(result['conf'][sequence_number]) > 50:
                (x, y, w, h) = (result['left'][sequence_number], result['top'][sequence_number], result['width'][sequence_number], result['height'][sequence_number])
                threshInv = cv2.rectangle(threshInv, (x, y), (x + w, y + h), (0, 255, 0), 2) #Draw text recognition area box
                game_words.append(result['text'][sequence_number].lower())


        # cv2.imshow('capturedText', threshInv)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #word_list = [word.lower() for word in result['text']]
        print(list(sorted(set(WORDS) & set(game_words), key = game_words.index)))
        return(list(sorted(set(WORDS) & set(game_words), key = game_words.index))) #sorts list by recognition order


# if __name__ == '__main__':
#         game = gameBoard()
#         game.getGameText(sys.argv[1])