#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-3-16 下午3:08
# @Author  : liuxinglong
# @File    : output_2d_gif.py.py
# @Description: blabla
from PIL import Image
import os
import pandas as pd
# import imageio
import shutil

input_path = "/home/liuxinglong/data/ISBI/isbi_preprocess_test/output2d"
# input_path = "/home/liuxinglong/data/ISBI/isbi_preprocess_test/output2dcomp"
labels_csv = "../data/labels_isbi_test.csv"
output_path = "/home/liuxinglong/data/ISBI/isbi_preprocess_test/output2dgif"


def generate_gif(path, image_series_name):
    idx_start = -1
    idx_end = -1

    images = []
    for idx in range(0, 200):
        image_name = os.path.join(path, "{}_{}.png".format(image_series_name, idx))
        # image_name = os.path.join(path, "{}_{}.jpg".format(image_series_name, idx))
        if os.path.isfile(image_name):
            if idx_start == -1:
                idx_start = idx
            else:
                idx_end = idx
            images.append(Image.open(image_name))
            # images.append(imageio.imread(image_name))

    image_length = idx_end - idx_start + 1
    abs_output_path = os.path.join(output_path, "{}.gif".format(image_series_name))
    # im = Image.open(abs_output_path)
    duration = 300.0
    print ("image length {}".format(image_length))
    # imageio.mimsave(abs_output_path, images, duration=duration, loop=1)

    first_frame_path = os.path.join(path, "{}_{}.png".format(image_series_name, idx_start))
    im = Image.open(first_frame_path)
    im.save(abs_output_path, save_all=True, append_images=images, loop=1, duration=duration)
    return


def main():
    data_csv = pd.read_csv(labels_csv)
    for idx, data_piece in data_csv.iterrows():
        image_name = data_piece["ID"]
        print ("processing {}".format(image_name))
        generate_gif(input_path, image_name)
    return


if __name__ == "__main__":
    main()






















