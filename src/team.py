import cv2
import numpy as np

from equipment import Equipment

class Team:
    def __init__(self):
        self.EquipmentImage = []
        self.Equipment = []
        self.EquipmentValue = 0

    def CalculateEquipmentValue(self):
        for e in self.Equipment:
            if e["count"] > 0:
                self.EquipmentValue += (int(e["cost"]) * int(e["count"]))

