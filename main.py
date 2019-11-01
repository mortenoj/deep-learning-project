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
    img_rgb = cv2.imread("./games/game3.png")
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    # template = cv2.imread("./icons/Image 102 at frame 0.png",0)
    template = cv2.imread("./icons/Image 125 at frame 0.png",0)

    w, h = template.shape[::-1]

    thresholdDist = 20
    detectedObjects=[]
    i = 0
    for scale in np.linspace(0.5, 2.0, 15)[::-1]:
        resized = imutils.resize(template, width = int(template.shape[1] * scale))
        res = cv2.matchTemplate(img_gray, resized, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)

        for pt in zip(*loc[::-1]):
            if len(detectedObjects) == 0 or notInList(pt, detectedObjects, thresholdDist):
                detectedObjects.append(pt)
                cellImage=img_rgb[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
                cv2.imwrite("results/"+str(pt[1])+"_"+str(pt[0])+".jpg",cellImage, 
                [int(cv2.IMWRITE_JPEG_QUALITY), 50])  
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
                i += 1

    print("Found", i , "matches")
    cv2.imwrite('res.png',img_rgb)


# application entry point
if __name__ == '__main__':
    main()