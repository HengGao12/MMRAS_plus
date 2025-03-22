# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
from PIL import Image, ImageDraw, ImageChops
import torch.nn.functional as F
from easydict import EasyDict
import scipy.stats as st

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
import tempfile
import random
import time
import warnings
import torch
import cv2
import numpy as np
import tqdm
import torch.nn as nn
import math

import seaborn as sns
import matplotlib.pylab as plt
from matplotlib import cm

import detectron2

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision import transforms
from mask2former import add_maskformer2_config, SemanticSegmentorWithTTA
from demo.predictor import VisualizationDemo
from torch.autograd import Variable
from matplotlib import pyplot
from torchvision.utils import save_image
import kornia as K

import ipdb

from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# constants
WINDOW_NAME = "mask2former demo"


def generalized_entropy(softmax_id_val, gamma=0.1, M=1):
    probs = softmax_id_val
    probs_sorted = torch.sort(probs, dim=1)[0][:, -M:]
    scores = torch.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma),
                        dim=1)

    return -scores


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="/home1/gaoheng/gh_workspace/Mask2Anomaly/configs/cityscapes/semantic-segmentation/anomaly_inference.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--input",
        default="/home1/gaoheng/gh_workspace/Mask2Anomaly/datasets/Validation_Dataset/FS_LostAndFound_full/images/*.png",
        # /home1/gaoheng/gh_workspace/Mask2Anomaly/datasets/Validation_Dataset/FS_LostAndFound_full/images/*.png
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        default="/home1/gaoheng/gh_workspace/Mask2Anomaly/anomaly_predictions/LostAndFound/",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--score",
        default="Mask2Anomaly",
        help="The Score Function for Calculating OOD Scores in [MSP, MaxLogits, EBO, GEN, Mask2Anomaly]."
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--aug",
        default=True,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    # GH added
    parser.add_argument(
        "--exp_name",
        help="Define the experiments' names",
        default="base-exp",
    )
    parser.add_argument(
        "--dataset",
        help="Define the dataset's name, choices: LostAndFound, RoadAnomaly",
        default="LostAndFound",
    )   
    parser.add_argument(
        "--text_enhance",
        type=bool,
        help="using text modal for enhancement or not",
        default=False,
    ) 
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    
    # load text embeddings
    text_embeddings_path = './pretrain/city_RN50_clip_text.pth'
    text_embeddings = torch.load(text_embeddings_path).cuda() # [19, 1024]
    # text_embeddings = text_embeddings.mean(dim=0).unsqueeze(0) # [1, 1024]
    # ipdb.set_trace()
    
    setup_logger(name="fvcore")
    logger = setup_logger(output='./results/' + args.dataset + '/' + args.exp_name + '.txt')
    logger.info("Arguments: " + str(args))
    
    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')
    
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    anomaly_score_list = []
    ood_gts_list = []

    # ipdb.set_trace()
    if args.input:
        # print(len(glob.glob(os.path.expanduser(str(args.input)))))
        # ipdb.set_trace()
        for path in glob.glob(os.path.expanduser(str(args.input))):
            img = read_image(path, format="BGR")
            start_time = time.time()
            
            img_ud = np.flipud(img) # Aug 1

            img_lr = np.fliplr(img) # Aug 2
            
            # ipdb.set_trace()
            
            predictions_na, _ = demo.run_on_image(img)
            # ipdb> for k in predictions_na.keys(): print(k)
            # sem_seg, feat (newly added key)
            predictions_lr, _ = demo.run_on_image(img_lr)

            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions_na["instances"]))
                    if "instances" in predictions_na
                    else "finished",
                    time.time() - start_time,
                )
            )
            

            
            
            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                    # ipdb.set_trace()
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output

                
                if args.score == "MSP":
                    predictions_naa =  predictions_na["sem_seg"].unsqueeze(0)
                    outputs_na = 1 - torch.max(predictions_naa[0:19,:,:], axis = 1)[0]
                    outputs_na = outputs_na.detach().cpu().numpy().squeeze().squeeze()
                    
                    predictions_lrr =  predictions_lr["sem_seg"].unsqueeze(0)
                    outputs_lr = 1 - torch.max(predictions_lrr[0:19,:,:], axis = 1)[0]
                    outputs_lr = outputs_lr.detach().cpu().numpy().squeeze().squeeze()
                    outputs_lr = np.flip(outputs_lr.squeeze(), 1)
                if args.score == "MaxLogits":
                    predictions_naa = predictions_na["sem_seg"].unsqueeze(0)
                    # get the original logits output
                    aprox_logits_na = torch.exp(torch.log(predictions_naa))
                    outputs_na, _ = torch.max(aprox_logits_na[0:19,:,:], dim=1)
                    outputs_na = 1 - outputs_na
                    outputs_na = outputs_na.detach().cpu().numpy().squeeze().squeeze()
                    
                    predictions_lrr = predictions_lr["sem_seg"].unsqueeze(0)
                    aprox_logits_lr = torch.exp(torch.log(predictions_lrr))
                    outputs_lr, _ = torch.max(aprox_logits_lr[0:19,:,:], dim=1)
                    outputs_lr = 1 - outputs_lr 
                    outputs_lr = outputs_lr.detach().cpu().numpy().squeeze().squeeze()
                    outputs_lr = np.flip(outputs_lr.squeeze(), 1)
                if args.score == "EBO":
                    predictions_naa = predictions_na["sem_seg"].unsqueeze(0)
                    # get the original logits output
                    aprox_logits_na = torch.exp(torch.log(predictions_naa))
                    outputs_na = 1.0 * torch.logsumexp(aprox_logits_na[0:19,:,:] / 1.0, 1)
                    outputs_na = 1 - outputs_na
                    outputs_na = outputs_na.detach().cpu().numpy().squeeze().squeeze()
                    
                    predictions_lrr = predictions_lr["sem_seg"].unsqueeze(0)
                    aprox_logits_lr = torch.exp(torch.log(predictions_lrr))
                    outputs_lr = 1.0 * torch.logsumexp(aprox_logits_lr[0:19,:,:] / 1.0, 1)
                    outputs_lr = 1 - outputs_lr 
                    outputs_lr = outputs_lr.detach().cpu().numpy().squeeze().squeeze()
                    outputs_lr = np.flip(outputs_lr.squeeze(), 1)
                if args.score == "GEN":
                    predictions_naa =  predictions_na["sem_seg"].unsqueeze(0)
                    outputs_na = generalized_entropy(predictions_naa[0:19,:,:])
                    outputs_na = outputs_na.detach().cpu().numpy().squeeze().squeeze()
                    
                    predictions_lrr =  predictions_lr["sem_seg"].unsqueeze(0)
                    outputs_lr = generalized_entropy(predictions_lrr[0:19,:,:])
                    outputs_lr = outputs_lr.detach().cpu().numpy().squeeze().squeeze()
                    outputs_lr = np.flip(outputs_lr.squeeze(), 1)                    
                if args.score == "test_score":
                    predictions_naa = predictions_na["sem_seg"].unsqueeze(0)
                    aprox_logits_na = torch.exp(torch.log(predictions_naa))
                    outputs_na = torch.exp(1-0.025 * torch.logsumexp(aprox_logits_na[0:19,:,:] / 0.025, 1))
                    outputs_na = outputs_na.detach().cpu().numpy().squeeze().squeeze()
                    
                    predictions_lrr = predictions_lr["sem_seg"].unsqueeze(0)
                    aprox_logits_lr = torch.exp(torch.log(predictions_lrr))
                    outputs_lr = torch.exp(1-0.025 * torch.logsumexp(aprox_logits_lr[0:19,:,:] / 0.025, 1))
                    outputs_lr = outputs_lr.detach().cpu().numpy().squeeze().squeeze()
                    outputs_lr = np.flip(outputs_lr.squeeze(), 1)
                if args.score == "Mask2Anomaly":
                    predictions_naa =  predictions_na["sem_seg"].unsqueeze(0)
                    
                    if args.text_enhance:
                        temp = F.conv2d(predictions_na["feat"], text_embeddings[:,:,None,None])
                        temp = F.interpolate(
                            temp,
                            size=(predictions_naa.shape[-2], predictions_naa.shape[-1]),
                            mode="bilinear",
                            align_corners=True,
                        )
                        predictions_naa[:,0:19,:,:] += 0.1*temp # strategy a: ehancing using add operation
                        
                    outputs_na = 1 - torch.max(predictions_naa[0:19,:,:], axis = 1)[0]  # original code

                    
                    # outputs_na = -(1 - torch.max(predictions_naa[0:19,:,:], axis = 1)[0])
                    if predictions_na["sem_seg"][19:,:,:].shape[0] > 1:
                        outputs_na_mask = torch.max(predictions_na["sem_seg"][19:,:,:].unsqueeze(0),  axis = 1)[0]
                        # GH added
                        # outputs_na_mask = torch.einsum('ck, ukv -> ukv', text_embeddings, outputs_na_mask)
                        
                        # temp = F.conv2d(predictions_na["sem_seg"][:19,:,:].unsqueeze(0), text_embeddings[:, :, None, None])
                        # ipdb.set_trace()
                        # outputs_na_mask[0,:19,:,:] = temp
                        outputs_na_mask[outputs_na_mask < 0.5] = 0
                        outputs_na_mask[outputs_na_mask >= 0.5] = 1
                        outputs_na_mask = 1 - outputs_na_mask # original code
                        
                        
                        # outputs_na_mask = -outputs_na_mask
                        outputs_na_save = outputs_na.clone().detach().cpu().numpy().squeeze().squeeze()
                        outputs_na = outputs_na*outputs_na_mask.detach()
                        outputs_na_mask = outputs_na_mask.detach().cpu().numpy().squeeze().squeeze()
                    outputs_na = outputs_na.detach().cpu().numpy().squeeze().squeeze()

                    #left-right
                    predictions_lrr =  predictions_lr["sem_seg"].unsqueeze(0)
                    
                    if args.text_enhance:
                        temp = F.conv2d(predictions_lr["feat"], text_embeddings[:,:,None,None])
                        temp = F.interpolate(
                            temp,
                            size=(predictions_lrr.shape[-2], predictions_naa.shape[-1]),
                            mode="bilinear",
                            align_corners=True,
                        )
                        predictions_lrr[:,0:19,:,:] += 0.1 * temp # strategy a: ehancing using add operation
                        
                    outputs_lr = 1 - torch.max(predictions_lrr[0:19,:,:], axis = 1)[0]  # original code
                    # outputs_lr = - torch.max(predictions_lrr[0:19,:,:], axis = 1)[0])
                    if predictions_lr["sem_seg"][19:,:,:].shape[0] > 1:
                        outputs_lr_mask = torch.max(predictions_lr["sem_seg"][19:,:,:].unsqueeze(0),  axis = 1)[0]
                        # GH added
                        # outputs_lr_mask[:19,:,:] = F.conv2d(predictions_lr["sem_seg"][:19,:,:], text_embeddings[:, :, None])
                        
                        outputs_lr_mask[outputs_lr_mask < 0.5] = 0
                        outputs_lr_mask[outputs_lr_mask >= 0.5] = 1
                        outputs_lr_mask = 1 - outputs_lr_mask # original code
                        
                        # outputs_lr_mask = -1 + outputs_lr_mask
                        outputs_lr_save = outputs_lr.clone()
                        outputs_lr = outputs_lr*outputs_lr_mask.detach()
                    outputs_lr = outputs_lr.detach().cpu().numpy().squeeze().squeeze()
                    outputs_lr = np.flip(outputs_lr.squeeze(), 1)

                # if args.score == "test_score":
                #     outputs = np.expand_dims(outputs_na, 0).astype(np.float32)
                # else:
                outputs = np.expand_dims((outputs_lr + outputs_na )/2.0, 0).astype(np.float32)
                pathGT = path.replace("images", "labels_masks")  
                
                # ipdb.set_trace()              

                if "RoadObsticle21" in pathGT:
                    pathGT = pathGT.replace("webp", "png")
                if "fs_static" in pathGT:
                    pathGT = pathGT.replace("jpg", "png")                
                if "RoadAnomaly" in pathGT:
                    pathGT = pathGT.replace("jpg", "png")  

                mask = Image.open(pathGT)
                ood_gts = np.array(mask)
                # ipdb.set_trace()

                if "RoadAnomaly" in pathGT:
                    ood_gts = np.where((ood_gts==2), 1, ood_gts)
                if "LostAndFound" in pathGT:
                    # np.unique [0, 1, 255]
                    ood_gts = np.where((ood_gts==0), 255, ood_gts)
                    ood_gts = np.where((ood_gts==1), 0, ood_gts)
                    ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

                    # ood_gts = np.where((ood_gts==1), 0, ood_gts)
                    # ood_gts = np.where((ood_gts==255), 1, ood_gts)
                    # ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
                    
                    # original
                    # ood_gts = np.where((ood_gts==1), 1, ood_gts)
                    # # ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)
                    # ood_gts = np.where((ood_gts==255), 0, ood_gts)
                    # ipdb.set_trace()
                if "fs_static" in pathGT:
                    # [0, 255]
                    ood_gts = np.where((ood_gts==255), 1, ood_gts)
                    
                if "Streethazard" in pathGT:
                    ood_gts = np.where((ood_gts==14), 255, ood_gts)
                    ood_gts = np.where((ood_gts<20), 0, ood_gts)
                    ood_gts = np.where((ood_gts==255), 1, ood_gts)

                if 1 not in np.unique(ood_gts):
                    continue              
                else:
                    ood_gts_list.append(np.expand_dims(ood_gts, 0))
                    anomaly_score_list.append(outputs)
                    # ipdb.set_trace()

    file.write( "\n")
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)
    # drop void pixels
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))

    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))
    
    # ipdb.set_trace()
    fpr, tpr, _ = roc_curve(val_label, val_out
                            # , pos_label=1
                            )    
    roc_auc = auc(fpr, tpr)
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    logger.info('AUROC score:{}, AUPRC score:{}, FPR@TPR95:{}.'.format(roc_auc, prc_auc, fpr))
    # print(f'AUPRC score: {prc_auc}')
    # print(f'FPR@TPR95: {fpr}')

    file.write(('AUROC score:' + str(roc_auc) + 'AUPRC score:' + str(prc_auc) + '   FPR@TPR95:' + str(fpr) ))
    file.close()