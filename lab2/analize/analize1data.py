import json

with open("../data/1analize.json", "r") as f:
    dt = json.loads(f.read())
print(dt)
