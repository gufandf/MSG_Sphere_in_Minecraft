import json
import cv2
from PIL import Image
from search import *
import math
import os, shutil
import subprocess
from multiprocessing import Pool
import re
from pathlib import Path
import time
from tqdm import tqdm

block = json.loads(open("./blocks.json", "r").read())
sored_block = []
MSG_center = (0.5, -32.5, 0.5)

last_blocks = {}
last_frame = 0
R = 114 / 2


def del_file(path):
    for elm in Path(path).glob("*"):
        # print(elm)
        elm.unlink() if elm.is_file() else shutil.rmtree(elm)

def tranImg(Path: str, Output: str):
    time_of_steps = {
        "check_block_pos": 0.0,
        "get_block": 0.0,
        "all_block_pos_append": 0.0,
        "function_append": 0.0,
        "end": 0.0,
    }
    last_frame = 0
    num = re.findall("\\d\\d\\d\\d\\d\\d", Path)
    if len(num) >= 1:
        if int(num[0]) > 1:
            last_frame = Image.open(f"./temp/frame{str(int(num[0])-1).zfill(6)}.png")
            last_frame_pixel = last_frame.load()
    img = Image.open(Path)
    img = img.resize((360, 180))
    pixels = img.load()
    all_block_pos = set()
    function = set()

    for i in range(360):
        for j in range(180):
            for k in range(1):
                pixel_value = pixels[i, j]  # type: ignore

                block_pos = (
                    math.floor(
                        MSG_center[0]
                        + R
                        * math.cos(math.radians(i + 90))
                        * math.cos(math.radians(j - 90))
                    ),
                    math.floor(MSG_center[1] - R * math.sin(math.radians(j - 90))),
                    math.floor(
                        MSG_center[2]
                        - R
                        * math.sin(math.radians(i + 90))
                        * math.cos(math.radians(j - 90))
                    ),
                )
                if block_pos in all_block_pos or block_pos[1] < -54:
                    break
                if not last_frame == 0:
                    last_frame_pixel_value = last_frame_pixel[i, j]  # type: ignore
                    distance = (
                        math.pow(last_frame_pixel_value[0] - pixel_value[0], 2)
                        + math.pow(last_frame_pixel_value[1] - pixel_value[1], 2)
                        + math.pow(last_frame_pixel_value[2] - pixel_value[2], 2)
                    )
                    if distance < 16:
                        break
                
                block = get_block(pixel_value)
                all_block_pos.add(block_pos)
                function.add("setblock"+ " "+ str(block_pos[0])+ " "+ str(block_pos[1])+ " "+ str(block_pos[2])+ " " + str(block))

    f = open(Output, "w", encoding="UTF-8")
    a = "\n".join(function)
    f.write(a)
    f.close()

    files = os.listdir("./temp")
    num_png = len(files)
    # print(f"\r已完成:{Path}")
    # print(time_of_steps)
    # time_of_steps["end"] += time.time() - start_time

def tranVideo(Path: str):
    del_file("./temp")
    del_file("./MSG Sphere/data/gufandf/function/msgsphere/frame")
    print("[TRANS MAIN]正在拆分帧")
    videoCapture = cv2.VideoCapture(Path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    subprocess.call(
        f'ffmpeg -loglevel quiet -i "{Path}" -vf "fps=20,scale=360:180" "./temp/frame%06d.png"'
    )

    # 获取总帧数
    total_frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    tick_count = math.ceil(total_frames/fps*20)
    
    # 线程数，CPU是几线程就改成多少
    pool = Pool(processes=15)

    print("[TRANS MAIN]等待所有线程完成...")
    n = 0

    # 使用tqdm创建进度条
    with tqdm(total=tick_count, desc="转换进度") as pbar:
        for path, floder, files in os.walk("./temp/"):
            for fileName in files:
                file_path = path + fileName
                result = pool.apply_async(
                    tranImg,
                    args=(
                        file_path,
                        f"./MSG Sphere/data/gufandf/function/msgsphere/frame/{fileName.split('.')[0]}.mcfunction",
                    ),
                    callback=lambda _: pbar.update(1)  # 每完成一个任务更新进度条
                )
                # tranImg(file_path,f"./MSG Sphere/data/gufandf/function/msgsphere/frame/{fileName.split('.')[0]}.mcfunction",)
                n += 1

        
        pool.close()
        pool.join()

    functions = []
    for i in range(n):
        functions.append(
            f"schedule function gufandf:msgsphere/frame/frame{str(i+1).zfill(6)} {i+1}t append"
        )
    function = "\n".join(functions)
    function += f"\nschedule function gufandf:msgsphere/function {n}t append"
    f = open(
        "./MSG Sphere/data/gufandf/function/msgsphere/function.mcfunction",
        "w",
        encoding="UTF-8",
    )
    f.write(function)
    f.close()

if __name__ == "__main__":
    start_time = time.time()
    # tranVideo("C:/Users/Gufandf/Videos/test_2s.mp4")
    # tranVideo("C:/Users/Gufandf/Documents/Adobe/Premiere Pro/25.0/MSGSphere/序列 06.mp4")
    # tranImg("./new/R-C.png","./MSG Sphere/data/gufandf/function/msgsphere/shehui.mcfunction",)
    # tranImg("./new/地球卫星地图贴图1.png","./MSG Sphere/data/gufandf/function/msgsphere/earth.mcfunction",)
    # tranVideo("C:/Users/Gufandf/Videos/眼睛.mp4")
    tranVideo("C:/Users/Gufandf/Videos/地球自转9s.mp4")
    # tranImg("./new/酷晨.png","./MSG Sphere/data/gufandf/function/msgsphere/kuchen.mcfunction")
    # tranImg("./new/一串神秘数字.png","./MSG Sphere/data/gufandf/function/msgsphere/num.mcfunction")
    # tranVideo("C:/Users/Gufandf/Videos/巨型球幕表情包_6.mp4")
    # tranVideo("C:/Users/Gufandf/Videos/kuchen.mp4")
    # tranVideo("C:/Users/Gufandf/Videos/牛肉sama.mp4")
    # tranVideo(r"C:\Users\Gufandf\Videos\13394367931214780.mp4")
    
    # tranVideo("C:/Users/Gufandf/Downloads/767832586-1-208.mp4")
    # tranImg("./new/木星表面.jpg","./MSG Sphere/data/gufandf/function/msgsphere/jupiter.mcfunction")
    # tranImg("C:/Users/Gufandf/Pictures/DayEnvironmentHDRI066_1K-TONEMAPPED.jpg","./MSG Sphere/data/gufandf/function/msgsphere/123.mcfunction")
    print(time.time() - start_time)
