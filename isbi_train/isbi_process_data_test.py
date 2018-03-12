#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 17:17:47 2018

@author: liuxinglong
"""
import os
import sys
import pandas as pd
import SimpleITK as sitk
import numpy as np

isbi_csv_file = "./data/labels_isbi_test.csv"

isbi_image_path = "/mnt/lustre/liuxinglong/data/ISBI/isbi_preprocess_test/image"
prep_folder = "../prep_result"
bbox_folder = "../bbox_result"


def debugmsg(msg):
    print msg


def _notice():
    print "*" * 60


def main():
    print "main running"

    label_csv = pd.read_csv(isbi_csv_file)
    target_spacing = [1.0, 1.0, 1.0]

    for label_piece in label_csv.iterrows():
        _notice()

        imagename = label_piece["ID"]
        token = imagename.find("_")
        imagepid = imagename[:token]
        imagetime = imagename[token + 1:]

        extend_box_id = "{}_{}_bbox.npy".format(imagepid, imagetime)
        target_processed_image_id = "{}_{}_clean.npy".format(imagepid, imagetime)
        label_id = "{}_{}".format(imagepid, imagetime)
        file_id = "{}_{}.mhd".format(imagepid, imagetime)
        bbox_id = "{}_{}_pbb.npy".format(imagepid, imagetime)
        lbb_id = "{}_{}_lbb.npy".format(imagepid, imagetime)
        debugmsg( "processing {}".format(label_id) )

        voxel_centroid = [0, 0, 0]
        try:
            voxel_centroid = np.array(eval(label_piece["Centroid"]))
        except Exception, e:
            debugmsg( "Eval Exception {}".format(e.message) )
            continue

        orig_image = sitk.ReadImage(os.path.join(isbi_image_path, file_id))
        orig_spacing = np.array(orig_image.GetSpacing())
        orig_shape = np.array(orig_image.GetSize())

        # target image loading
        target_processed_image_array = np.load(os.path.join(prep_folder, target_processed_image_id))
        target_processed_image_array = target_processed_image_array[0]
        target_processed_image = sitk.GetImageFromArray(target_processed_image_array)

        ##########################################
        # extend_box is Z prior !
        ##########################################
        extend_box = np.load(os.path.join(prep_folder, extend_box_id))
        debugmsg( extend_box )

        target_voxel_centroid = voxel_centroid * (orig_spacing / target_spacing)
        target_shape = orig_shape * (orig_spacing / target_spacing)
        debugmsg( "voxel centroid {} -> {}".format(voxel_centroid, target_voxel_centroid) )
        debugmsg( "shape {} -> {}".format(orig_shape, target_shape) )

        final_voxel_centroid = target_voxel_centroid - np.array([extend_box[2][0], extend_box[1][0], extend_box[0][0]])
        debugmsg( "final voxel centroid {}".format(final_voxel_centroid) )

        orig_voxels = 0
        try:
            orig_voxels = np.array(label_piece["Voxels"])
        except Exception, e:
            debugmsg( "Eval Exception {}".format(e.message) )
            continue

        target_size = pow((3 * float(orig_voxels) / (4.0 * np.pi)), 1.0 / 3) * np.max(orig_spacing / target_spacing)
        debugmsg( "final target size {}".format(target_size) )

        # sitk.Show(target_processed_image)
        ##########################################
        # also flipped!!!
        ##########################################
        target_pbb = np.array([[1.0, final_voxel_centroid[2], final_voxel_centroid[1], final_voxel_centroid[0], target_size]])
        target_bbox_name = os.path.join(bbox_folder, bbox_id)

        target_lbb = np.array([[final_voxel_centroid[2], final_voxel_centroid[1], final_voxel_centroid[0], target_size]])
        target_lbb_name = os.path.join(bbox_folder, lbb_id)

        np.save(target_bbox_name, target_pbb)
        np.save(target_lbb_name, target_lbb)


if __name__ == "__main__":
    main()



