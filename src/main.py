import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import imutils

def notInList(newObject, detectedObjects, thresholdDist = 20):
    for detectedObject in detectedObjects:
        if math.hypot(newObject[0]-detectedObject[0],newObject[1]-detectedObject[1]) < thresholdDist:
            return False
    return True

def main():
    img_rgb = cv2.imread("../public/games/game4.png")
    template = cv2.imread("../public/icons/ak.png", 0)

    matches = FindMatches(img_rgb, template, 0.8, np.linspace(0.5, 2.0, 15)[::-1], True)

    print(matches, " matches found")


def FindMatches(img_rgb, template, threshold, linspace, debug = False):
    w, h = template.shape[::-1]
    img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)


    detectedObjects=[]

    matches = 0
    for scale in linspace:
        resized = imutils.resize(template, width = int(template.shape[1] * scale))
        res = cv2.matchTemplate(img, resized, cv2.TM_CCOEFF_NORMED)
        loc = np.where( res >= threshold)

        for pt in zip(*loc[::-1]):
            if len(detectedObjects) == 0 or notInList(pt, detectedObjects):
                detectedObjects.append(pt)
                if debug:
                    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
                matches += 1
    
    if debug:
        cv2.imwrite('res.png', img_rgb)
            
    return matches

# application entry point
if __name__ == '__main__':
    main()
