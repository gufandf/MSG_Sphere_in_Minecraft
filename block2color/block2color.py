import cv2
import os
import json


def get_block_color(id:str):
    path = f"./block2color/blockstates/{id}.json"
    models = get_block_models(path)
    imgs = []
    for model in models:
        model_path = f"./block2color/models/block/{model}.json"
        imgs += get_model_imgs(model_path)
    colors = []
    for img in imgs:
        img_path = f"./block2color/block/{img}.png"
        colors.append(get_img_color(img_path))
    color = [0,0,0]
    for i in colors:
        color[0] += i[0]
        color[1] += i[1]
        color[2] += i[2]
    color[0] //= len(colors)
    color[1] //= len(colors)
    color[2] //= len(colors)
    return color


def get_block_models(Path:str):
    block = json.loads(open(Path,"r").read())
    models = []
    for key1 in block:
        for key2 in block[key1]:
            if type(block[key1][key2]) is list:
                for i in block[key1][key2]:
                    models.append(i["model"].split("/")[-1])
            else:    
                models.append(block[key1][key2]["model"].split("/")[-1])
    return models

def get_model_imgs(Path:str):
    imgs = []
    model = json.loads(open(Path,"r").read())
    for key in model["textures"]:
        img = model["textures"][key]
        imgs.append(img.split("/")[-1])
    return imgs


#     colors.append(get_img_color(img))
# color = [0,0,0]
# for i in colors:
#     color[0] += i[0]
#     color[1] += i[1]
#     color[2] += i[2]
# color[0] /= len(colors)
# color[1] /= len(colors)
# color[2] /= len(colors)

def get_img_color(Path:str):
    image = cv2.imread(Path)
    average_color_per_channel = cv2.mean(image)
    average_color = (int(average_color_per_channel[2]), int(average_color_per_channel[1]), int(average_color_per_channel[0]))
    return average_color


if __name__ == "__main__":
    blockToColor = {}
    for files in os.walk("./block2color/blockstates/"):
        for file in files[2]:
            # print(file)
            color = get_block_color(file.split(".")[0])
            blockToColor[file.split(".")[0]] = color
    f = open("./block2color/block2color.json","w",encoding="UTF-8")
    f.write(json.dumps(blockToColor))