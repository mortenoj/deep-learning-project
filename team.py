from equipment import Equipment

class Team:
    def __init__(self, side, inventory):
        self.side = side
        self.inventory = inventory
        self.totalWorth

    def CalculateTotalWorth(self):
        self.totalWorth = 0
        for equipment in self.inventory:
            self.totalWorth += equipment.worth
        
        return self.totalWorth