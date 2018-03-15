#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-3-15 下午2:09
# @Author  : liuxinglong
# @File    : output_2d_slice.py
# @Description: blabla
import SimpleITK as sitk
import pandas as pd
import os
import sys
import numpy as np
import cv2

labels_csv = "../data/labels_isbi_test.csv"
data_src = "/home/liuxinglong/data/ISBI/isbi_preprocess_test/forreview"
output_path = "/home/liuxinglong/data/ISBI/isbi_preprocess_test/output2d"

def lung_trans(img):
    """
    transform the img using given threshold
    :param img: input and output image
    :note : python doesn't support direct ref call, the input
    image is actually the reference of the variable in caller
    :return: No return
    """
    lungwin = np.array([-1200., 600.])
    img = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    img[img < 0] = 0
    img[img > 1] = 1
    img = (img * 255).astype('uint8')
    return img


def output_image(img, centroid, diameter, path):
    img_arr = sitk.GetArrayFromImage(img)
    depth, width, height = img_arr.shape
    img_arr = lung_trans(img_arr)

    slice_center_z = centroid[2]
    slice_center_y = centroid[1]
    slice_center_x = centroid[0]
    # make sure this is int since cv2.circle() can only accept integer radiuses
    radius_scale = 2.0
    MAX_OUTPUT_R = 96

    slice_radius = int(radius_scale * diameter)
    slice_center = centroid[2]
    tmp_slice = np.zeros([MAX_OUTPUT_R * 2, MAX_OUTPUT_R * 2]).astype("uint8")
    for slice_idx in xrange(max(0, slice_center - slice_radius), min(slice_center + slice_radius, depth)):
        img_slice = img_arr[slice_idx, ...]
        rgbslice = cv2.cvtColor(img_slice, cv2.COLOR_GRAY2RGB)

        # in-place drawing
        cv2.circle(
            rgbslice, (centroid[0], centroid[1]), slice_radius, (0, 0, 255), 1)

    # for slice_idx in xrange(max(0, slice_center_z - slice_radius), min(slice_center_z + slice_radius, depth)):
    #     left = max(0, slice_center_y - MAX_OUTPUT_R)
    #     right = min(slice_center_y + MAX_OUTPUT_R, width)
    #     top = max(0, slice_center_x - MAX_OUTPUT_R)
    #     down = min(slice_center_x + MAX_OUTPUT_R, height)
    #
    #     # print "outputting image crop [({0}, {1}),({2}, {3})]".format(left, right, top, down)
    #
    #     deltaX = right - left
    #     deltaY = down - top
    #     tmp_slice[0:deltaX, 0:deltaY] = img_arr[slice_idx, left:right, top:down]
    #     rgbslice = cv2.cvtColor(tmp_slice, cv2.COLOR_GRAY2RGB)
    #
        cv2.imwrite("{}_{}.png".format(path, slice_idx), rgbslice)


def main():
    data = pd.read_csv(labels_csv)

    for idx, data_piece in data.iterrows():
        image_id = data_piece["ID"]
        centroid = data_piece["Centroid"]
        voxels = data_piece["Voxels"]
        radius = pow((3 * float(voxels ) / (4.0 * np.pi)), 1.0 / 3)

        print "processing {}".format(image_id)

        try:
            centroid = eval(centroid)
        except Exception, e:
            print "eval wrong {}".format(e.message)

        image_path = os.path.join(data_src, "{}.mhd".format(image_id))
        # mask_path = os.path.join(data_src, "{}_mask.mhd".format(image_id))
        image = sitk.ReadImage(image_path)
        # mask = sitk.ReadImage(mask_path)

        output_image(image, centroid, radius * 2, os.path.join(output_path, image_id))


if __name__ == "__main__":
    main()




