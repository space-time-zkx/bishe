# import os, sys
#
# import cv2
#
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# TOOL_DIR = os.path.dirname(BASE_DIR)
# ROOT_DIR = os.path.dirname(TOOL_DIR)
# sys.path.append(ROOT_DIR)
# import numpy as np
# import argparse
# import tqdm
# from PIL import Image
# from lib.utils.kitti.calibration import Calibration
#
# data_root = os.path.join(ROOT_DIR, '/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/object')
# parser = argparse.ArgumentParser()
# parser.add_argument('--gen_train', action='store_true', help='Generate train split pseudo lidar [train and val]')
# parser.add_argument('--gen_test', action='store_true', help='Generate test split pseudo lidar')
# parser.add_argument('--sampling', action='store_true', help='sample dense points or not')
# parser.add_argument('--sampling_rate', type=float, default=0.8, help='number of samples')
# parser.add_argument('--vis_test', action='store_true',
#                     help='visulize pesudo lidar [default case: 000001.bin in training set]')
# args = parser.parse_args()
# import torch
# def read_calib_file(filepath):
#     data = {}
#     with open(filepath, "r") as f:
#         for line in f.readlines():
#             line = line.rstrip()
#             if len(line) == 0:
#                 continue
#             key, value = line.split(":", 1)
#             try:
#                 data[key] = np.array([float(x) for x in value.split()])
#             except ValueError:
#                 pass
#
#     return data
# def dynamic_baseline(calib_info):
#     P3 = np.reshape(calib_info["P3"], [3, 4])
#     P = np.reshape(calib_info["P2"], [3, 4])
#     baseline = P3[0, 3] / (-P3[0, 0]) - P[0, 3] / (-P[0, 0])
#     return baseline
# def disp2depth(disp, calib):
#     depth = calib / disp.clamp(min=1e-8)
#     return depth
# def depth2points(tag, total_files):
#     # img_path_prefix = os.path.join(data_root, tag, '09_26_0014')
#     # depth_path_prefix = os.path.join(data_root, tag, 'depth_psmnet')
#     img_path_prefix = "/data/zhaokexin/vision/kitti_frame/2011_09_26/2011_09_26_drive_0014_sync/image_02/data"
#     depth_path_prefix ="/data/zhaokexin/vision/disp_estimate/PSMNet-master/09_26_00142"
#     calib_path_prefix = os.path.join(data_root, tag, 'calib')
#     output_path_prefix = os.path.join(data_root, tag, '09_26_0014')
#
#     progress_bar = tqdm.tqdm(total=total_files, leave=True, desc='%s data generation' % tag)
#     testlist = list(np.loadtxt("/data/zhaokexin/vision/kitti_frame/09_26_0014.txt"))
#     # for i in range(total_files):
#     for i in testlist:
#         print(i)
#         depth_path = os.path.join(depth_path_prefix, '%010d.png' % i)
#         if os.path.exists(depth_path):
#             # calib_path = os.path.join(calib_path_prefix, '%06d.txt' % i)
#             calib_path = "/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/object/training/09_26_0014.txt"
#             output_path = os.path.join(output_path_prefix, '%010d.bin' % i)
#
#             depth = np.array(Image.open(depth_path)).astype(np.float32)
#             depth = depth / 256
#
#             # depth = np.loadtxt(depth_path)
#             calib = Calibration(calib_path)
#             calib_info = read_calib_file(calib_path)
#             calib1 = np.reshape(calib_info["P2"], [3, 4])[0, 0] * dynamic_baseline(calib_info)
#             height, width = depth.shape
#             uvdepth = np.zeros((height, width, 3), dtype=np.float32)
#             depth = disp2depth(torch.from_numpy(depth), calib1).cpu().numpy()
#             cv2.imwrite("depth.png",depth)
#             # add RGB values
#             img_path = os.path.join(img_path_prefix, '%010d.png' % i)
#             img = np.array(Image.open(img_path)).astype(np.float32)
#
#             imback = np.zeros((depth.shape[0],depth.shape[1],3))
#
#             imback[:img.shape[0],:img.shape[1],:]=img
#
#             img = imback
#             print(img.shape,uvdepth.shape)
#             assert img.shape == uvdepth.shape, 'RGB image and depth map should have the same shape'
#             uvdepth = np.concatenate((uvdepth, img), axis=2)
#
#             for v in range(height):
#                 for u in range(width):
#                     uvdepth[v, u, 0] = u
#                     uvdepth[v, u, 1] = v
#             uvdepth[:, :, 2] = depth
#
#             uvdepth = uvdepth.reshape(-1, 6)
#
#             # sampling, to reduce the number of pseudo lidar points
#             # if args.sampling:
#             #     num_points = uvdepth.shape[0]
#             #     choice = np.random.choice(num_points, int(num_points * args.sampling_rate), replace=True)
#             #     uvdepth = uvdepth[choice]
#
#             points = calib.img_to_rect(uvdepth[:, 0], uvdepth[:, 1], uvdepth[:, 2])
#             points = calib.rect_to_lidar(points)
#             points = np.concatenate((points,np.ones((len(points),1))),-1)
#             # print(points.shape)
#             points = np.concatenate((points,img.reshape(-1,3)),-1)
#             # points = np.concatenate((points, uvdepth[:, 3:6]), -1)
#
#             # remove points with heights larger than 1.0 meter
#             idx = np.argwhere(points[:, 2] <= 1.0)
#             # print(np.max(points[:, 0]),np.max(points[:,1]),np.max(points[:,2]))
#             # idx = np.argwhere(np.logical_and(points[:, 2] <= 0,abs(points[:,1])<20))
#             points = points[idx, :].squeeze(1)
#
#             points.tofile(output_path)
#             # np.savetxt("1295points.txt",points)
#         progress_bar.update()
#         break
#     progress_bar.close()
#
#
# def vis_demo():
#     import mayavi.mlab as mlab
#     # data_root = "/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/object/training/09_26_0014"
#     # pseudo_lidar_path = os.path.join(data_root, 'testing', 'pseudo_lidar', '000000.bin')
#     pseudo_lidar_path = "/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/object/training/09_26_0014/0000000046.bin"
#     # pseudo_lidar_path ="/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/object/training/pseudo_lidar/000000.bin"
#     print(pseudo_lidar_path)
#     assert os.path.exists(pseudo_lidar_path)
#     # channels = 6 if args.add_rgb else 3
#     channels = 4
#     pesudo_lidar = np.fromfile(pseudo_lidar_path).reshape(-1, channels)
#     print(pesudo_lidar)
#     idx = np.argwhere(np.logical_and(pesudo_lidar[:, 0] >0,abs(pesudo_lidar[:,1])<20))
#     pesudo_lidar = pesudo_lidar[idx, :].squeeze(1)
#     # print(pesudo_lidar.shape)
#     fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
#     color = pesudo_lidar[:, 2]
#
#     mlab.points3d(pesudo_lidar[:, 0], pesudo_lidar[:, 1], pesudo_lidar[:, 2], color*2, color=None,
#                   mode='point', colormap='plasma', scale_factor=1, figure=fig)
#
#     # draw origin
#     mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)
#
#     # draw axis
#     # axes = np.array([
#     #     [5., 0., 0., 0.],
#     #     [0., 5., 0., 0.],
#     #     [0., 0., 5., 0.],
#     # ], dtype=np.float64)
#     # mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
#     # mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
#     # mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)
#
#     # draw fov
#     # fov = np.array([  # 45 degree
#     #     [10., 10., 0., 0.],
#     #     [10., -10., 0., 0.],
#     # ], dtype=np.float64)
#     # mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
#     # mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
#
#     mlab.view(azimuth=180, elevation=75, focalpoint=[ 20.0909996 , -1.04700089, -2.03249991], distance=120.0, figure=fig)
#     mlab.savefig('pc_view.jpg', figure=fig)
#     mlab.show()
#     # input()
#
#
# if __name__ == '__main__':
#     if args.gen_train:
#         depth2points(tag='training',
#                      total_files=1)
#
#     if args.gen_test:
#         depth2points(tag='testing',
#                      total_files=7518)
#
#     if args.vis_test:
#         vis_demo()
#
#     # vis_demo()
#
import os, sys

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(TOOL_DIR)
sys.path.append(ROOT_DIR)
import numpy as np
import argparse
import tqdm
from PIL import Image
from lib.utils.kitti.calibration import Calibration

data_root = os.path.join(ROOT_DIR, '/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/object')
parser = argparse.ArgumentParser()
parser.add_argument('--gen_train', action='store_true', help='Generate train split pseudo lidar [train and val]')
parser.add_argument('--gen_test', action='store_true', help='Generate test split pseudo lidar')
parser.add_argument('--sampling', action='store_true', help='sample dense points or not')
parser.add_argument('--sampling_rate', type=float, default=0.8, help='number of samples')
parser.add_argument('--vis_test', action='store_true',
                    help='visulize pesudo lidar [default case: 000001.bin in training set]')
args = parser.parse_args()
import torch


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


def disp2depth(disp, calib):
    depth = calib / disp.clamp(min=1e-8)
    return depth


def depth2points(tag, total_files):
    img_path_prefix = os.path.join(data_root, tag, 'image_2')
    # depth_path_prefix = os.path.join(data_root, tag, 'depth_psmnet')
    img_path_prefix = "/data/zhaokexin/vision/kitti_frame/2011_09_28/2011_09_28_drive_0037_sync/image_02/data"
    depth_path_prefix = "/data/zhaokexin/vision/disp_estimate/PSMNet-master/09_28_00372"
    calib_path_prefix = os.path.join(data_root, tag, 'calib')
    output_path_prefix = os.path.join(data_root, tag, '09_28_00372')
    # output_path_prefix = "/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/object/training/09_26_0014"
    progress_bar = tqdm.tqdm(total=total_files, leave=True, desc='%s data generation' % tag)
    # testlist = list(np.loadtxt("/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/ImageSets/test.txt"))
    testlist = list(np.loadtxt("/data/zhaokexin/vision/kitti_frame/09_28_0037.txt"))

    for i in testlist:
        # i = 4926
        # print(i)
        depth_path = os.path.join(depth_path_prefix, '%010d.png' % i)
        # print(depth_path)
        if os.path.exists(depth_path):
            calib_path = os.path.join(calib_path_prefix, '%06d.txt' % 4614)
            # calib_path = "/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/object/training/09_26_0014.txt"
            output_path = os.path.join(output_path_prefix, '%010d.bin' % i)

            depth = np.array(Image.open(depth_path)).astype(np.float32)
            depth = depth / 256

            # depth = np.loadtxt(depth_path)
            calib = Calibration(calib_path)
            calib_info = read_calib_file(calib_path)
            calib1 = np.reshape(calib_info["P2"], [3, 4])[0, 0] * dynamic_baseline(calib_info)
            height, width = depth.shape
            uvdepth = np.zeros((height, width, 3), dtype=np.float32)
            # depth = disp2depth(torch.from_numpy(depth), calib1).cpu().numpy()
            depth = calib1/depth
            where_are_inf = np.isinf(depth)

            # nan替换成0,inf替换成nan
            depth[where_are_inf] = 0
            # cv2.imwrite("depth.png",depth)
            # print(depth,np.max(depth),np.min(depth))
            # add RGB values
            img_path = os.path.join(img_path_prefix, '%010d.png' % i)
            img = np.array(Image.open(img_path)).astype(np.float32)
            assert img.shape == uvdepth.shape, 'RGB image and depth map should have the same shape'
            uvdepth = np.concatenate((uvdepth, img), axis=2)

            for v in range(height):
                for u in range(width):
                    uvdepth[v, u, 0] = u
                    uvdepth[v, u, 1] = v
            uvdepth[:, :, 2] = depth

            uvdepth = uvdepth.reshape(-1, 6)

            # sampling, to reduce the number of pseudo lidar points
            if args.sampling:
                num_points = uvdepth.shape[0]
                choice = np.random.choice(num_points, int(num_points * args.sampling_rate), replace=True)
                uvdepth = uvdepth[choice]

            points = calib.img_to_rect(uvdepth[:, 0], uvdepth[:, 1], uvdepth[:, 2])
            points = calib.rect_to_lidar(points)
            points = np.concatenate((points, np.ones((len(points), 1))), -1)
            # print(points.shape)
            points = np.concatenate((points,img.reshape(-1,3)),-1)
            # points = np.concatenate((points, uvdepth[:, 3:6]), -1)

            # remove points with heights larger than 1.0 meter

            # idx = np.argwhere(points[:, 2] <= 1.0)
            # print(np.max(points[:, 0]),np.max(points[:,1]),np.max(points[:,2]))
            idx = np.argwhere(np.logical_and(points[:, 2] <= 1,uvdepth[:, 2]!=0))

            points = points[idx, :].squeeze(1)
            # print(points.shape, depth.shape)
            points.tofile(output_path)
            # np.savetxt("points1.txt",points)
        progress_bar.update()
        # break
    progress_bar.close()


def vis_demo():
    import mayavi.mlab as mlab
    # data_root = "/data/zhaokexin/dataset/KITTI_3dobject/object"
    # pseudo_lidar_path = os.path.join(data_root, 'testing', 'pseudo_lidar', '000000.bin')
    pseudo_lidar_path= "/data/zhaokexin/vision/3d_object/patchnet-master/data/KITTI/object/training/09_26_0014/0000000046.bin"
    print(pseudo_lidar_path)
    assert os.path.exists(pseudo_lidar_path)
    # channels = 6 if args.add_rgb else 3
    channels = 7
    pesudo_lidar = np.fromfile(pseudo_lidar_path).reshape(-1, channels)
    print(pesudo_lidar)
    # np.savetxt("cloud.txt",pesudo_lidar)
    idx = np.argwhere(np.logical_and(pesudo_lidar[:, 0] > 0, abs(pesudo_lidar[:, 1]) < 20))
    pesudo_lidar = pesudo_lidar[idx, :].squeeze(1)
    # print(pesudo_lidar.shape)
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1000, 500))
    color = pesudo_lidar[:, 2]

    mlab.points3d(pesudo_lidar[:, 0], pesudo_lidar[:, 1], pesudo_lidar[:, 2], color * 2, color=None,
                  mode='point', colormap='plasma', scale_factor=1, figure=fig)

    # draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.2)

    # draw axis
    # axes = np.array([
    #     [5., 0., 0., 0.],
    #     [0., 5., 0., 0.],
    #     [0., 0., 5., 0.],
    # ], dtype=np.float64)
    # mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig)
    # mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig)
    # mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig)

    # draw fov
    # fov = np.array([  # 45 degree
    #     [10., 10., 0., 0.],
    #     [10., -10., 0., 0.],
    # ], dtype=np.float64)
    # mlab.plot3d([0, fov[0,0]], [0, fov[0,1]], [0, fov[0,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)
    # mlab.plot3d([0, fov[1,0]], [0, fov[1,1]], [0, fov[1,2]], color=(1,1,1), tube_radius=None, line_width=1, figure=fig)

    mlab.view(azimuth=180, elevation=75, focalpoint=[20.0909996, -1.04700089, -2.03249991], distance=120.0, figure=fig)
    mlab.savefig('pc_view.jpg', figure=fig)
    mlab.show()
    # input()


if __name__ == '__main__':
    if args.gen_train:
        depth2points(tag='training',
                     total_files=1)

    if args.gen_test:
        depth2points(tag='testing',
                     total_files=7518)

    if args.vis_test:
        vis_demo()

    # vis_demo()

