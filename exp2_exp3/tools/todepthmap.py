import numpy as np
import torch
import os
import cv2
def disp2depth(disp, calib):
    depth = calib / disp.clamp(min=1e-8)
    return depth
def read_calib_file(filepath):
    data = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(":", 1)
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data
def dynamic_baseline(calib_info):
    P3 = np.reshape(calib_info["P3"], [3, 4])
    P = np.reshape(calib_info["P2"], [3, 4])
    baseline = P3[0, 3] / (-P3[0, 0]) - P[0, 3] / (-P[0, 0])
    return baseline
calibpath ="/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/object/training/calib"
disppath = "/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/object/training/testsubdisp"
for file in os.listdir(disppath):
    # if not os.path.exists("/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/object/training/mydepth/"+file.replace("txt","png")):
    calib= read_calib_file(os.path.join(calibpath,file))
    calib1 = np.reshape(calib["P2"], [3, 4])[0, 0] * dynamic_baseline(calib)
    depth = np.loadtxt(os.path.join(disppath,file))
    depth = disp2depth(torch.from_numpy(depth).cuda(), torch.tensor(calib1).cuda()).detach().cpu().numpy()
    np.savetxt("/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/object/training/mydepth2/"+file,depth)
        # cv2.imwrite("/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/object/training/mydepth/"+file.replace("txt","png"),depth)