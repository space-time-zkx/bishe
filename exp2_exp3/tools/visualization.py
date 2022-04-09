import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
import cv2
from PIL import Image
import numpy as np
from lib.utils.kitti.kitti_utils import get_objects_from_label
from lib.utils.kitti.calibration import Calibration


def get_image(img_file):
    assert os.path.exists(img_file)
    return Image.open(img_file)  # (H, W, 3) RGB mode

def get_label(label_file):
    assert os.path.exists(label_file)
    return get_objects_from_label(label_file)

def get_calib(calib_file):
    assert os.path.exists(calib_file)
    return Calibration(calib_file)

def get_lidar(lidar_file):
    assert os.path.exists(lidar_file)
    return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
def get_pseudu_lidar_rgb(lidar_file):
    assert os.path.exists(lidar_file)
    return np.loadtxt(lidar_file).reshape(-1, 6)



def draw_projected_box3d(image, corners3d, color=(255, 255, 255), thickness=2):
    '''
    draw 3d bounding box in image plane
    input:
        image: RGB image
        corners3d: (8,3) array of vertices (in image plane) for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''

    corners3d = corners3d.astype(np.int32)
    # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness)

        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (corners3d[i, 0], corners3d[i, 1]), (corners3d[j, 0], corners3d[j, 1]), color, thickness)

    return image



def draw_lidar_points(pc, color=None, fig=None, bgcolor=(0,0,0), pts_scale=2, pts_mode='point'):
    if pc.shape[0]==0:
        return fig
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    t = np.linspace(0, 2 * np.pi, len(pc))
    # x = np.sin(2 * t)
    # y = np.cos(t)
    # z = np.cos(2 * t)

    # Create a [0..len(t)) index that we'll pass as 's'
    s = np.arange(len(t))

    # Create and populate lookup table (the integer index in s corresponding
    #   to the point will be used as the row in the lookup table
    lut = np.zeros((len(s), 4))

    # A simple lookup table that transitions from red (at index 0) to
    #   blue (at index len(data)-1)
    
    # p3d = mlab.points3d(x, y, z, s, scale_mode='none')
    # p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(s)
    # p3d.module_manager.scalar_lut_manager.lut.table = lut
    if color is None:
        x1 = pc[:, 0]  # x position of point
        x = pc [:,0]
        y = pc [:,1]
        z = pc [:,2]
        y1 = pc[:, 1]  # y position of point
        col = np.sqrt(x1 ** 2 + y ** 2)  # map distance
        # print(col,type(col))

    else:
        col = color
    # print(len(col))
    if pts_mode=='sphere':
        mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color=col, mode='sphere', scale_factor=0.1, figure=fig)
    else:
        # color1 = []
        
        mlab.points3d(pc[:,0], pc[:,1], pc[:,2],  col, mode='point', colormap='spectral', scale_factor=pts_scale, figure=fig)
        
        # for i in range(len(pc)):
        #     # print(tuple(pc[i,3:]/255))
        #     color1.append(tuple(pc[i,3:]/255))
        # for row in s:
        #     f = (row/len(s))
        #     lut[row,:] = [pc[row,3],pc[row,4],pc[row,5],255]
        # print("ok", lut[0])
        # p3d = mlab.points3d(x, y, z,s  ,mode='point',scale_mode='none', scale_factor=1 ,figure=fig)
        # p3d.module_manager.scalar_lut_manager.lut.number_of_colors = len(s)
        # p3d.module_manager.scalar_lut_manager.lut.table = lut
       
       
        # mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color = col, mode='point', colormap='spectral', scale_factor=pts_scale, figure=fig)
        # mlab.points3d(pc[:,0], pc[:,1], pc[:,2],color=col, mode='point', colormap='spectral', scale_factor=pts_scale,figure=fig)
        # mlab.draw()
        # mlab.show()
    # draw origin point
    mlab.points3d(0, 0, 0, color=(1,1,1), mode='sphere', scale_factor=0.2)

    # draw axis
    axes=np.array([
        [1.,0.,0.,0.],
        [0.,1.,0.,0.],
        [0.,0.,1.,0.],
    ],dtype=np.float64)
    mlab.plot3d([0, axes[0,0]], [0, axes[0,1]], [0, axes[0,2]], color=(1,0,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1,0]], [0, axes[1,1]], [0, axes[1,2]], color=(0,1,0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2,0]], [0, axes[2,1]], [0, axes[2,2]], color=(0,0,1), tube_radius=None, figure=fig)

    return fig


def draw_lidar_3dbox(box3d, fig, color=(255,255,0), line_width=2, draw_text=True, text_scale=(1, 1, 1), color_list=None):
    num = len(box3d)
    for n in range(num):
        b = box3d[n]
        if color_list is not None:
            color = color_list[n]
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    return fig




if __name__ == '__main__':
    # loading image (No.000000 in training split)
    # img_file = '../resources/image.png'
    img_file = "../data/KITTI/object/training/image_2/001295.png"
    img = get_image(img_file)

    # loading objects (ground truth or predicted boxes)
    # obj_file = '../resources/label.txt'
    obj_file = "../data/KITTI/object/training/label_2/001295.txt"
    objs = get_label(obj_file)

    # loading calib
    # calib_file = '../resources/calib.txt'
    calib_file = "../data/KITTI/object/training/calib/001295.txt"
    calib = get_calib(calib_file)

    # loading lidar points
    # lidar_file = '../resources/lidar.bin'
    # lidar_file = "../data/KITTI/object/training/velodyne/000035.bin"
    # points = get_lidar(lidar_file)
    points = np.loadtxt("./data_prepare/1295points.txt")


    ########### visualize bbox in image plane ############
    # PIL to cv2 for the requirments of drawing function
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # generate 3d bbox in image plane
    corners3d = np.zeros((len(objs), 8, 3), dtype=np.float32)  # N * 8 * 3
    for i in range(len(objs)):
        corners3d[i] = objs[i].generate_corners3d()  # generate corners in 3D rect space
    
    _, box3ds = calib.corners3d_to_img_boxes(corners3d) # project corners from 3D space to image plane

    # draw 3d bbox
    id = 0
    for box3d in box3ds:
        if objs[id].cls_type=="Car":
            img = draw_projected_box3d(img, box3d, color=(0,255,0), thickness=1)
        elif objs[id].cls_type=="Pedestrian":
            # pass
            img = draw_projected_box3d(img, box3d, color=(255,255,0), thickness=1)
        elif objs[id].cls_type=="Cyclist":
            img = draw_projected_box3d(img, box3d, color=(0,255,255), thickness=1)
        id+=1

    # show and save
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # cv2 to PIL
    # img.show()
    img.save('../resources/001295_bbox_in_image_plane.png')



    ########### visualize bbox in 3D world sapce  #############
    # NOTE: if you want to use the following codes, you need to install mayavi
    # refer to: https://github.com/charlesq34/frustum-pointnets/blob/master/mayavi/mayavi_install.sh
    # and please comment the following codes if you didn't install mayavi

    import mayavi.mlab as mlab
    fig = mlab.figure(size=(1200, 800), bgcolor=(0.9, 0.9, 0.9))
    # draw lidar points
    fig = draw_lidar_points(points, fig=fig)
    corners3d = calib.rect_to_lidar(corners3d.reshape(-1, 3))  # from rect coordinate system to lidar coordinate system
    corners3d = corners3d.reshape(-1, 8, 3)
    for i in range(len(objs)):
        if objs[i].cls_type=="Car":
            fig = draw_lidar_3dbox(np.array([corners3d[i]]), fig=fig, color=(0, 1, 0))
        elif objs[i].cls_type=="Pedestrian":
            # pass
            fig = draw_lidar_3dbox(np.array([corners3d[i]]), fig=fig, color=(0, 1, 1))
        elif objs[i].cls_type=="Cyclist":
            fig = draw_lidar_3dbox(np.array([corners3d[i]]), fig=fig, color=(1, 1, 0))
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991], distance=62.0, figure=fig)
    mlab.show()
    mlab.savefig('../resources/001295_bbox_in_3d_space.png', figure=fig)
    # input()

