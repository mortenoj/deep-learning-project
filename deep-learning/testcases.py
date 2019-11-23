

def filter_testcase(testcase, match_round):
    if testcase == 1: # Pistol Rounds
        if match_round["CTSideTotalWorth"] < 5000 and match_round["TSideTotalWorth"] < 5000:
            return True
    elif testcase == 2: # Rounds that are not pistol rounds
        if match_round["CTSideTotalWorth"] > 5000 and match_round["TSideTotalWorth"] > 5000:
            return True
    elif testcase == 3: # Full-buy rounds
        if match_round["CTSideTotalWorth"] > 20500 and match_round["TSideTotalWorth"] > 18500:
            return True
    elif testcase == 4: # Fullbuy CT vs Full-Eco T-Side
        if match_round["CTSideTotalWorth"] > 20500 and match_round["TSideTotalWorth"] < 5000:
            return True
    elif testcase == 5: # Fullbuy T vs Full-Eco CT-Side
        if match_round["CTSideTotalWorth"] < 5000 and match_round["TSideTotalWorth"] > 18500:
            return True
    elif testcase == 6: # One side has an AWP
        for item in match_round["CTEquipment"]:
            if item["Name"] == "AWP":
                if item["Count"] > 0:
                    return True
        for item in match_round["TEquipment"]:
            if item["Name"] == "AWP":
                if item["Count"] > 0:
                    return True
    elif testcase == 7: # CT Side has kit
        for item in match_round["CTEquipment"]:
            if item["Name"] == "Defuse Kit":
                if item["Count"] > 0:
                    return True
    elif testcase == 8: # Fullbuy T vs Half-Investment CT-Side
        if match_round["CTSideTotalWorth"] < 15000 and match_round["CTSideTotalWorth"] > 7500 and match_round["TSideTotalWorth"] > 18500:
            return True
    elif testcase == 9: # Fullbuy CT vs Half-Investment T-Side
        if match_round["TSideTotalWorth"] < 15000 and match_round["TSideTotalWorth"] > 7500 and match_round["CTSideTotalWorth"] > 20500:
            return True
    else:
        return False
    
    return False