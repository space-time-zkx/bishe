import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import cv2
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V



class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
            
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

from collections import defaultdict
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder
import torch.utils.data as torch_data
class MyDataTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True,logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names

        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

        if hasattr(self.data_processor, "depth_downsample_factor"):
            self.depth_downsample_factor = self.data_processor.depth_downsample_factor
        else:
            self.depth_downsample_factor = None

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def prepare_data(self, data_dict):
        if data_dict.get('points', None) is not None:
            data_dict = self.point_feature_encoder.forward(data_dict)

        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )
        return data_dict
    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}

        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                elif key in ['gt_boxes2d']:
                    max_boxes = 0
                    max_boxes = max([len(x) for x in val])
                    batch_boxes2d = np.zeros((batch_size, max_boxes, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        if val[k].size > 0:
                            batch_boxes2d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_boxes2d
                elif key in ["images", "depth_maps"]:
                    # Get largest image size (H, W)
                    max_h = 0
                    max_w = 0
                    for image in val:
                        max_h = max(max_h, image.shape[0])
                        max_w = max(max_w, image.shape[1])

                    # Change size of images
                    images = []
                    for image in val:
                        pad_h = common_utils.get_pad_params(desired_size=max_h, cur_size=image.shape[0])
                        pad_w = common_utils.get_pad_params(desired_size=max_w, cur_size=image.shape[1])
                        pad_width = (pad_h, pad_w)
                        # Pad with nan, to be replaced later in the pipeline.
                        pad_value = np.nan

                        if key == "images":
                            pad_width = (pad_h, pad_w, (0, 0))
                        elif key == "depth_maps":
                            pad_width = (pad_h, pad_w)

                        image_pad = np.pad(image,
                                           pad_width=pad_width,
                                           mode='constant',
                                           constant_values=pad_value)

                        images.append(image_pad)
                    ret[key] = np.stack(images, axis=0)
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size
        return ret
class Mydata(MyDataTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, data_path=None, root_path=None, chindex="000000",logger=None,):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training,logger=logger
        )
        self.root_path=root_path
        self.data_path=data_path
        self.chindex=chindex
    def __getitem__(self,key):
        
        points = np.fromfile(self.data_path+"/"+self.chindex+".bin", dtype=np.float32).reshape(-1, 4)
        input_dict={
            'points': points,
            'frame_id': int(self.chindex),
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/pv_rcnn.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default="../checkpoints/pv_rcnn_8369.pth", help='specify the pretrained model')
    parser.add_argument('--root_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--index', type=str, default='-1', help='specify the extension of your point cloud data file')
    parser.add_argument('--mayavi', action="store_true",default=False, help='specify the extension of your point cloud data file')
    parser.add_argument('--save_path',type=str,default="", help='specify the extension of your point cloud data file')
    parser.add_argument('--conf',type=float,default=0.5, help='specify the extension of your point cloud data file')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    root_path = args.root_path
    data_path = root_path+"/velodyne"
    maya = args.mayavi
    save_path = args.save_path 
    mlab.options.offscreen = True
    if maya:
        mlab.options.offscreen = False
    chindex=args.index
    conf = args.conf
    if chindex==-1:
        
        demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(data_path), ext=args.ext, logger=logger
        )
        logger.info(f'Total number of samples: \t{len(demo_dataset)}')
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
        with torch.no_grad():
            for idx, data_dict in enumerate(demo_dataset):
                logger.info(f'Visualized sample index: \t{idx + 1}')
                data_dict = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)
                p = data_dict['points']
                V.draw_scenes(
                    points=p[:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                    ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                    conf=conf
                )
                bboxs = V.boxes_to_corners_3d(pred_dicts[0]['pred_boxes'].cpu().numpy())
                pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
                pred_cls = pred_dicts[0]['pred_labels'].cpu().numpy()
                num_bboxs = len(bboxs)
                ones = np.array([[[1]*8]*num_bboxs])
                right = np.transpose(ones,(1,2,0))
                # print(bboxs.shape,right.shape)
                bboxs = np.concatenate((bboxs,right),axis=2)
                ind = str(int(data_dict['frame_id'][0])).zfill(6)
                bboxcorners = V.velodyne2img(root_path+"/calib",ind,bboxs)
                colors = []
                for c in list(pred_cls):
                    if c==2:
                        colors.append((255,255,0))
                    elif c==1:
                        colors.append((0,255,0))
                    else:
                        colors.append((0,255,255))
                image = V.draw_projected_box3d(cv2.imread(root_path+"/image_2/"+ind+".png"),bboxcorners,pred_scores,pred_cls,colors,conf=conf)
                # print(image)
                cv2.imwrite(save_path+"/"+ind+".png",image)
                mlab.savefig(save_path+"/"+ind+"_1.png")
                if maya:
                    mlab.show(stop=True)
    else:
        mydata = Mydata(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,data_path=data_path, root_path=root_path, chindex=chindex,logger=logger)
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=mydata)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()
        with torch.no_grad():
            data_dict = mydata[0]
            data_dict = mydata.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            p = data_dict['points']
            V.draw_scenes(
                points=p[:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                conf=conf
            )
            bboxs = V.boxes_to_corners_3d(pred_dicts[0]['pred_boxes'].cpu().numpy())
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
            pred_cls = pred_dicts[0]['pred_labels'].cpu().numpy()
            num_bboxs = len(bboxs)
            ones = np.array([[[1]*8]*num_bboxs])
            right = np.transpose(ones,(1,2,0))
            # print(bboxs.shape,right.shape)
            bboxs = np.concatenate((bboxs,right),axis=2)
            ind = chindex
            bboxcorners = V.velodyne2img(root_path+"/calib",ind,bboxs)
            # print(pred_cls)
            colors = []
            for c in list(pred_cls):
                if c==2:
                    colors.append((255,255,0))
                elif c==1:
                    colors.append((0,255,0))
                else:
                    colors.append((0,255,255))
            image = V.draw_projected_box3d(cv2.imread(root_path+"/image_2/"+ind+".png"),bboxcorners,pred_scores,pred_cls,colors,conf=conf)
            # print(image)
            cv2.imwrite(save_path+"/"+ind+".png",image)
            mlab.savefig(save_path+"/"+ind+"_1.png")
            if maya:
                mlab.show(stop=True)
    logger.info('Demo done.')


if __name__ == '__main__':
    main()
