import sys

import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import base64
import io
from imageio import imread

from words import WORDS, two_word_mapping


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

    return sorted(list(set(WORDS) & set(mapped_words)), key=mapped_words.index)


class gameBoard:
    def __init__(self):
        pass

    def getGameText(self, imgEncoding, use_b64_encoding=True):
        if use_b64_encoding:
            img = imread(io.BytesIO(base64.b64decode(imgEncoding)))
        else:
            img = cv2.imread(imgEncoding,cv2.IMREAD_COLOR)
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

        return codenamify_words(game_words)


if __name__ == '__main__':
    # game = gameBoard()
    # words = game.getGameText(sys.argv[1], False)
    # print(words, len(words))

    word_list = ["ice", "loch", "ness", "ice", "cream", "new", "york"]
    print(codenamify_words(word_list))
