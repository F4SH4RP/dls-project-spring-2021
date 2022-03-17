import telebot
from PIL import Image
import torch

import os
import tkinter.filedialog
import torch, torchvision
import mmdet
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# Choose to use a config and initialize the detector
config = 'configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.py'
# Setup a checkpoint file to load
checkpoint = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
# initialize the detector
model = init_detector(config, checkpoint, device='cpu')

def detect():
    img = tkinter.filedialog.askopenfilename()
    result = inference_detector(model, img)
    img2 = show_result_pyplot(model, img, result, score_thr=0.3)
detect() 
