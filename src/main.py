import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import math
import json
from PIL import Image

from team import Team

def notInList(newObject, detectedObjects, thresholdDist = 25):
    for detectedObject in detectedObjects:
        if math.hypot(newObject[0]-detectedObject[0],newObject[1]-detectedObject[1]) < thresholdDist:
            return False
    return True

def main():
    global EquipmentList

    gameImage = cv2.imread("../public/games/game1.png")
    ctIsOnLeft = True

    leftImg = gameImage[500:900, 0:550] # left
    rightImg = np.fliplr(gameImage[500:900, 1370:1920]) # right

    ct = Team()
    t = Team()

    # cv2.imshow("cropped", leftImg)
    # cv2.waitKey(0)
    # cv2.imshow("cropped", rightImg)
    # cv2.waitKey(0)

    if ctIsOnLeft:
        ct.Equipment = GetTeamEquipment(leftImg)
        t.Equipment = GetTeamEquipment(rightImg)
    else:
        ct.Equipment = GetTeamEquipment(rightImg)
        t.Equipment = GetTeamEquipment(leftImg)

    ct.CalculateEquipmentValue()
    t.CalculateEquipmentValue()

    print("CT Equipment list")
    for e in ct.Equipment:
        print(e["name"], e["count"])

    print("CT side equipment value: ", ct.EquipmentValue)
    # print("T side equipment value: ", t.EquipmentValue)

def GetTeamEquipment(equipmentImage):
    with open("../public/icons/_data.json", 'r') as f:
        equipmentList = json.load(f)

    cv2.imwrite('res.png', equipmentImage)
    for elem in equipmentList:
        elem["count"] = FindEquipmentInImage(equipmentImage, elem, True)

    return equipmentList

def FindEquipmentInImage(image, equipment, debug = False):
    template = cv2.imread("../public/" + equipment["path"], 0)
    w, h = template.shape[::-1]

    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    linspace = np.linspace(0.8, 2.2, 15)[::-1] 
    threshold = 0.8 

    debugImg = cv2.imread("res.png")

    if equipment["type"] == "pistol":
        linspace = np.linspace(0.2, 0.8, 5)[::-1] 
        threshold = 0.942
    elif equipment["type"] == "utility":
        linspace = np.linspace(0.2, 1.0, 15)[::-1]
        threshold = 0.92
    elif equipment["type"] == "rifle":
        linspace = np.linspace(1.0, 2.0, 10)[::-1] 
        threshold = 0.75 
    elif equipment["type"] == "smg":
        linspace = np.linspace(0.8, 2.0, 10)[::-1] 

    rects = []
    debugType = "rifle"

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
                    rects.append(pt)
                matches += 1
    
    if debug:
        for pt in rects:
            cv2.rectangle(debugImg, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        cv2.imwrite('res.png', debugImg)
    return matches

# application entry point
if __name__ == '__main__':
    main()
