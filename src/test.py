import glob
import json

data = []

for file in glob.glob("icons/*.png"):
    if "_dark" in file:
        continue

    name = file.replace("icons/", "").replace(".png", "")
    data.append({
        'name': name,
        'path': file,
        'type': '',
        'cost': 0,
    })


with open('data.json', 'w') as outfile:
    json.dump(data, outfile)

