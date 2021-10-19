import sys

import cv2
import pytesseract
from pytesseract import Output

from words import WORDS

if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])

    #Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Threshold image (convert to binary). All pixels with value greater than T (70) are set to 255 (white). Otherwise, set to black (0).
    (T, threshInv) = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

    cv2.imshow("Threshold Binary Inverse", threshInv)

    cv2.waitKey(0)

    cv2.destroyAllWindows()

    #psm 6 assumes a single block of text
    #oem 3 uses default engine mode
    custom_config = r'--oem 3 --psm 6'

    result = pytesseract.image_to_data(threshInv, output_type=Output.DICT,  config=custom_config, lang='eng')

    total_boxes = len(result['text'])
    for sequence_number in range(total_boxes):
        #Check confidence score
        if int(result['conf'][sequence_number]) > 50:
            (x, y, w, h) = (result['left'][sequence_number], result['top'][sequence_number], result['width'][sequence_number], result['height'][sequence_number])
            threshInv = cv2.rectangle(threshInv, (x, y), (x + w, y + h), (0, 255, 0), 2) #Draw text recognition area box

    cv2.imshow('capturedText', threshInv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    word_list = [word.lower() for word in result['text']]
    print(list(sorted(set(WORDS) & set(word_list), key = word_list.index))) #sorts list by recognition order
