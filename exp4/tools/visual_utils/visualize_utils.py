import mayavi.mlab as mlab
import numpy as np
import torch
import os
import cv2
box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]

def velodyne2img(calib_dir, img_id, velo_box):
    """
    :param calib_dir: calib文件的地址
    :param img_id: 要转化的图像id
    :param velo_box: (n,8,4)，要转化的velodyne frame下的坐标，n个3D框，每个框的8个顶点，每个点的坐标（x,y,z,1）
    :return: (n,4)，转化到 image frame 后的 2D框 的 x1y1x2y2
    """
    # 读取转换矩阵
    calib_txt=os.path.join(calib_dir, img_id) + '.txt'
    calib_lines = [line.rstrip('\n') for line in open(calib_txt, 'r')]
    for calib_line in calib_lines:
        if 'P2' in calib_line:
            P2=calib_line.split(' ')[1:]
            P2=np.array(P2, dtype='float').reshape(3,4)
        elif 'R0_rect' in calib_line:
            R0_rect=np.zeros((4,4))
            R0=calib_line.split(' ')[1:]
            R0 = np.array(R0, dtype='float').reshape(3, 3)
            R0_rect[:3,:3]=R0
            R0_rect[-1,-1]=1
        elif 'velo_to_cam' in calib_line:
            velo_to_cam = np.zeros((4, 4))
            velo2cam=calib_line.split(' ')[1:]
            velo2cam = np.array(velo2cam, dtype='float').reshape(3, 4)
            velo_to_cam[:3,:]=velo2cam
            velo_to_cam[-1,-1]=1
 
    tran_mat=P2.dot(R0_rect).dot(velo_to_cam)  # 3x4
 
    velo_box=velo_box.reshape(-1,4).T
    img_box = np.dot(tran_mat, velo_box).T
    img_box=img_box.reshape(-1,8,3)
 
    img_box[:,:,0]=img_box[:,:,0]/img_box[:,:,2]
    img_box[:, :, 1] = img_box[:, :, 1] / img_box[:, :, 2]
    img_box=img_box[:,:,:2]   # （n,8,2）
 
    # x1y1=np.min(img_box,axis=1)
    # x2y2 = np.max(img_box, axis=1)
    # result =np.hstack((x1y1,x2y2))   #（n,4）
 
    return img_box

def draw_projected_box3d(image, qs, pred_scores,pred_cls,color=(0, 255, 0),conf=0.3, thickness=1):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    height = image.shape[0]
    width = image.shape[1]
    qs = qs.astype(np.int32)
    for n in range(len(qs)):
        bbox = qs[n]
        bbox = [b for b in bbox if b[0]<width and b[0]>0 and b[1]<height and b[1]>0]
        x1y1 = np.min(qs[n],axis=0)
        if len(bbox)==8 and pred_scores[n]>=conf:
            for k in range(0, 4):
                # print(color[n],pred_cls[n])
                # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
                i, j = k, (k + 1) % 4
                # use LINE_AA for opencv3
                # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
                cv2.line(image, (qs[n, i, 0], qs[n,i, 1]), (qs[n,j, 0], qs[n,j, 1]), color[n], thickness)
                i, j = k + 4, (k + 1) % 4 + 4
                cv2.line(image, (qs[n,i, 0], qs[n,i, 1]), (qs[n,j, 0], qs[n,j, 1]), color[n], thickness)

                i, j = k, k + 4
                cv2.line(image, (qs[n,i, 0], qs[n,i, 1]), (qs[n,j, 0], qs[n,j, 1]), color[n], thickness)
                cv2.putText(image,str(round(pred_scores[n],2)),(int(x1y1[0]),int(x1y1[1])),cv2.FONT_HERSHEY_SIMPLEX,0.6,color=color[n],thickness=1)
    return image



def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig


def draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(600, 600))

    if isinstance(color, np.ndarray) and color.shape[0] == 1:
        color = color[0]
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    if isinstance(color, np.ndarray):
        pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
        pts_color[:, 0:3] = color
        pts_color[:, 3] = 255
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], np.arange(0, pts_color.__len__()), mode='sphere',
                          scale_factor=scale_factor, figure=fig)
        G.glyph.color_mode = 'color_by_scalar'
        G.glyph.scale_mode = 'scale_by_vector'
        G.module_manager.scalar_lut_manager.lut.table = pts_color
    else:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere', color=color,
                      colormap='gnuplot', scale_factor=scale_factor, figure=fig)

    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
    mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), line_width=3, tube_radius=None, figure=fig)

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None,conf=0.3):
    if not isinstance(points, np.ndarray):
        points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    fig = visualize_pts(points)
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0 :
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100,conf=conf)
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                mask = (ref_labels == k)
                fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=100,conf=conf)
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig


def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None, conf=0.3):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    import mayavi.mlab as mlab
    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)
        if cls[n]>=conf:
            if cls is not None:
                if isinstance(cls, np.ndarray):
                    mlab.text3d(b[6, 0], b[6, 1]+0.5, b[6, 2]+0.5, '%.2f' % cls[n], scale=(1, 1, 1), color=color, figure=fig)
                else:
                    mlab.text3d(b[6, 0], b[6, 1]+0.5, b[6, 2]+0.5, '%s' % cls[n], scale=(1, 1, 1), color=color, figure=fig)

            for k in range(0, 4):
                i, j = k, (k + 1) % 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                            line_width=line_width, figure=fig)

                i, j = k + 4, (k + 1) % 4 + 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                            line_width=line_width, figure=fig)

                i, j = k, k + 4
                mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                            line_width=line_width, figure=fig)

            i, j = 0, 5
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)
            i, j = 1, 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

    return fig
