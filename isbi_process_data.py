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

isbi_csv_file = "labels_isbi.csv"
isbi_src_path = "/home/liuxinglong/data/ISBI/isbi_preprocess/single"
prep_folder = "./prep_result"
bbox_folder = "./bbox_result"

IDs = ["1001","1003","1004","1005","1006","1007","1008","1009","1010","1011","1012","1014","1015","1016","1017","1021","1022","1024","1028","1031","1032","1034","1040","1045","1048","1050","1051","1052","1064","1066"]

items = ["19990102", "20000102"]

def debugmsg(msg):
    print msg


def _notice():
    print "*" * 60

def main():
    print "main running"

    label_csv = pd.read_csv(isbi_csv_file)
    target_spacing = [1.0, 1.0, 1.0]

    for thisid in IDs:
        for item in items:
            _notice()

            extend_box_id = "{}_{}_bbox.npy".format(thisid, item)
            target_processed_image_id = "{}_{}_clean.npy".format(thisid, item)
            label_id = "{}_Y{}".format(thisid, item)
            file_id = "{}_Y{}.mhd".format(thisid, item)
            bbox_id = "{}_{}_pbb.npy".format(thisid, item)
            lbb_id = "{}_{}_lbb.npy".format(thisid, item)
            debugmsg( "processing {}".format(label_id) )
            
            this_csv_piece = label_csv[label_csv["ID"] == label_id]
            voxel_centroid = [0, 0, 0]
            try:
                voxel_centroid = np.array(eval(this_csv_piece["Centroid"].item()))
            except Exception, e:
                debugmsg( "Eval Exception {}".format(e.message) )
                continue

            orig_image = sitk.ReadImage(os.path.join(isbi_src_path, file_id))
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
                orig_voxels = np.array(this_csv_piece["Voxels"].item())
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



