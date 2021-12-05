import sys

import cv2
import numpy as np
import base64
import io
from imageio import imread
import pandas
import random as rng
from PIL import Image
from statistics import mode, mean, median
from queue import PriorityQueue

from words import ALL_WORDS

class colorCard:
    def __init__(self):
        pass

    def getColorName(self, R,G,B):
        minimum = 10000
        #define colors to match known colorcard pixel values
        colors = [[210, 37, 51],[217, 200, 155],[0, 120, 221], [24, 37, 57]]
        color_names = ["red", "tan", "blue", "black"]

        #determine minimum distance and assign corresponding color name
        for i in range(len(colors)):
            color = colors[i]
            d = abs(R- color[0]) + abs(G- color[1])+ abs(B- color[2])
            if(d<=minimum):
                minimum = d
                cname = color_names[i]
        return cname

    def matchInnerRatios(self, contourRatios, k):

        # Make a max heap of difference with
        # first k elements.
        innerGrid = PriorityQueue()
        length = len(contourRatios)
        medianRatio = median(contourRatios)
        for i in range(k):
            innerGrid.put((-abs(contourRatios[i]-medianRatio),i))

        # Now process remaining elements
        for i in range(k,length):
            diff = abs(contourRatios[i]-medianRatio)
            ratio,rank = innerGrid.get()
            curr = -ratio

            # If difference with current
            # element is more than root,
            # then put it back.
            if diff>curr:
                innerGrid.put((-curr,rank))
                continue
            else:

                # Else remove root and insert
                innerGrid.put((-diff,i))

        final = []
        while(not innerGrid.empty()):
            ratio,rank = innerGrid.get()
            final.append(contourRatios[rank])

        return final


    def getColorCode(self, imgEncoding):
        #img= cv2.imread(imgEncoding,cv2.IMREAD_COLOR)
        img = imread(io.BytesIO(base64.b64decode(imgEncoding)))
        img = cv2.cvtColor(img, cv2.IMREAD_ANYCOLOR)

        color_pattern = []
        output = ""


        #convert image to RGB pixel values for color identification
        cimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        #apply adaptive threshold
        threshInv = cv2.adaptiveThreshold(blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 1)

        #find contours
        orig_contours, hierarchy = cv2.findContours(threshInv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        maxArea = max([cv2.contourArea(x) for x in orig_contours])

        grid_ratios = []
        for c in orig_contours:
            area = cv2.contourArea(c)
            grid_ratio = area / float(maxArea)
            if((grid_ratio >= 0.01 and grid_ratio <= .03)):
                grid_ratios.append(grid_ratio)


        inner_grid = self.matchInnerRatios(grid_ratios, 25)

        contours = []
        xcoords = []
        ycoords = []

        #isolate inner grid and rank contour coordinates by quantile
        for i,c in enumerate(orig_contours):
            area = cv2.contourArea(c)
            grid_ratio = area / float(maxArea)
            if(grid_ratio in inner_grid):
                contours.append(orig_contours[i])
                (x,y),_ = cv2.minEnclosingCircle(c)
                xcoords.append(int(x))
                ycoords.append(int(y))

        try:
            yranks = pandas.qcut(ycoords,5,labels=[1, 2, 3, 4, 5])
            xranks = pandas.qcut(xcoords,5,labels=[1, 2, 3, 4, 5])
        except ValueError:
            return []



        #sort contours
        contours, xcoords, ycoords = zip(*sorted(zip(contours, xranks, yranks), key = lambda b:[b[2], b[1]], reverse=False))

        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)

        #correct and approximate contour shape, determine bounding rect coordinates
        for i, c in enumerate(contours):
            hull = cv2.convexHull(c)
            contours_poly[i] = cv2.approxPolyDP(hull, 4, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])

        drawing = np.zeros((threshInv.shape[0], threshInv.shape[1], 3), dtype=np.uint8)


        #Draw bonding rects
        for i,c in enumerate(contours):
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
                (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)

            avg_color = cv2.mean(cimg[int(boundRect[i][1]): int(boundRect[i][1]+boundRect[i][3]), int(boundRect[i][0]): int(boundRect[i][0]+boundRect[i][2])])
            label = self.getColorName(avg_color[0], avg_color[1], avg_color[2])
            color_pattern.append(label)

            cv2.putText(drawing, str(i), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 2, cv2.LINE_AA)


        return(color_pattern)

if __name__ == '__main__':
        card = colorCard()
        card.getColorCode(sys.argv[1])
