import json
import pandas as pd
import numpy as np

f = open("./block2color/block2color.json","r",encoding="UTF-8")
blockToColor = json.loads(f.read())
blockToColor = pd.Series(blockToColor)
# f = open("./color2block.json","r",encoding="UTF-8")
# color2block = json.loads(f.read())
f.close()

def get_color(block:str):
    print(block)
    return blockToColor.get(block, default=(0,0,0))


def get_block(color:tuple) -> str:
    min = 196608
    target_block = ""
    for key,value in blockToColor.items():
        d = (value[0]-color[0])**2+ (value[1]-color[1])**2+ (value[2]-color[2])**2
        if d < min:
            min = d
            target_block = key
    # print(target_block)
    return str(target_block)