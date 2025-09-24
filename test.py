import numpy as np
import torch
import os
import traceback
import time
import nrrd
import sys
import matplotlib.pyplot as plt
import logging
import argparse
import torch.nn.functional as F
import SimpleITK as sitk
from scipy.stats import norm
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel
from scipy.ndimage.measurements import label
from scipy.ndimage import center_of_mass
from net.nodule_net import NoduleNet
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.bbox_reader import BboxReader
# from dataset.mask_reader import MaskReader
from config import config, train_config
from utils.visualize import draw_gt, draw_pred, generate_image_anim
from utils.util import dice_score_seperate, get_contours_from_masks, merge_contours, hausdorff_distance
from utils.util import onehot2multi_mask, normalize, pad2factor, crop_boxes2mask_single, npy2submission
import pandas as pd
from evaluationScript.noduleCADEvaluationLUNA16 import noduleCADEvaluation

plt.rcParams['figure.figsize'] = (9, 6)
plt.switch_backend('agg')
this_module = sys.modules[__name__]
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


parser = argparse.ArgumentParser()
parser.add_argument('--net', '-m', metavar='NET', default=config['net'],
                    help='neural net')
parser.add_argument("mode", type=str,
                    help="you want to test or val")
parser.add_argument("--weight", type=str, default=config['initial_checkpoint'],
                    help="path to model weights to be used")
parser.add_argument("--dicom-path", type=str, default=None,
                    help="path to dicom files of patient")
parser.add_argument("--out-dir", type=str, default=config['out_dir'],
                    help="path to save the results")
parser.add_argument("--test-set-name", type=str, default=config['test_set_name'],
                    help="path to save the results")
parser.add_argument("--fold-num", type=str, default='1',
                    help="fold-num")
parser.add_argument('--data-dir', default=train_config['DATA_DIR'], type=str, metavar='OUT',
                    help='path to load data')
parser.add_argument('--pvalue', default=0, type=int, metavar='OUT',
                    help='path to load data')


def main():
    logging.basicConfig(format='[%(levelname)s][%(asctime)s] %(message)s', level=logging.INFO)
    args = parser.parse_args()
    # params_eye_L = np.load('weights/params_eye_L.npy').item()
    # params_eye_R = np.load('weights/params_eye_R.npy').item()
    # params_brain_stem = np.load('weights/params_brain_stem.npy').item()

    if args.mode == 'eval':
        data_dir = args.data_dir
        test_set_name = {'B60f':'/root/workspace/imgctl/filelists/5_cross_val_filesets_seed2/filelistB60f_test_fold'+args.fold_num+'.npy',
                      'B31f':'/root/workspace/imgctl/filelists/5_cross_val_filesets_seed2/filelistB31f_test_fold'+args.fold_num+'.npy'}
        num_workers = 2
        initial_checkpoint = args.weight
        net = args.net
        out_dir = args.out_dir

        net = getattr(this_module, net)(config)
        net = net.cuda()

        if initial_checkpoint:
            print('[Loading model from %s]' % initial_checkpoint)
            checkpoint = torch.load(initial_checkpoint)
            # out_dir = checkpoint['out_dir']
            epoch = checkpoint['epoch']

            net.load_state_dict(checkpoint['state_dict'])
        else:
            print('No model weight file specified')
            return

        print('out_dir', out_dir)
        save_dir = os.path.join(out_dir, 'res', str(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(os.path.join(save_dir, 'FROC')):
            os.makedirs(os.path.join(save_dir, 'FROC'))
        dataset = BboxReader(data_dir,test_set_name,config, mode='eval')
#         dataset = BboxReader(data_dir, test_set_name, config, mode='eval')
#         dataset = MaskReader(data_dir, test_set_name, config, mode='eval')
        eval(net, dataset, save_dir)
        noduleCADEval(dataset, save_dir,epoch,out_dir,args.fold_num,args.pvalue)
    else:
        logging.error('Mode %s is not supported' % (args.mode))


def eval(net, dataset, save_dir=None):
    net.set_mode('eval')
    net.use_detect = True
    net.use_rcnn = True
#     raw_dir = config['data_dir']
#     preprocessed_dir = config['preprocessed_data_dir']

    print('Total # of eval data %d' % (len(dataset)))
    for i, (b60f_input, b31f_input, truth_bboxes, truth_labels, image) in enumerate(dataset):
        try:
            D, H, W = image.shape
            pid = dataset.b60_filenames[i]

            print('[%d] Predicting %s' % (i, pid), image.shape)

            with torch.no_grad():
                b60f_input = b60f_input.cuda().unsqueeze(0)
#                 fake_input = torch.zeros(input.shape).cuda()
                b31f_input = b31f_input.cuda().unsqueeze(0)
                net.forward(b31f_input,b60f_input, truth_bboxes, truth_labels)

            rpns = net.rpn_proposals.cpu().numpy()
            detections = net.detections.cpu().numpy()
            ensembles = net.ensemble_proposals.cpu().numpy()

            print('rpn', rpns.shape)
            print('detection', detections.shape)
            print('ensemble', ensembles.shape)

#             if pid == '0005974155_20191207_BCB60f':
#                 patch_dir = os.path.join(save_dir, 'patch')
#                 if not os.path.exists(patch_dir):
#                     os.makedirs(patch_dir)
#                 b60_patch = net.b60_inputs.cpu().numpy()
#                 b31_patch = net.b31_inputs.cpu().numpy()
#                 residual_patch = net.ic_out.cpu().numpy()
#                 generate_patch = net.generate_patch.cpu().numpy()
#                 np.save(os.path.join(patch_dir, '%s_b60_patch.npy' % (pid)), b60_patch)
#                 np.save(os.path.join(patch_dir, '%s_b31_patch.npy' % (pid)), b31_patch)
#                 np.save(os.path.join(patch_dir, '%s_residual_patch.npy' % (pid)), residual_patch)
#                 np.save(os.path.join(patch_dir, '%s_generate_patch.npy' % (pid)), generate_patch)
            if len(rpns):
                rpns = rpns[:, 1:]
                np.save(os.path.join(save_dir, '%s_rpns.npy' % (pid)), rpns)

            if len(detections):
                detections = detections[:, 1:-1]
                np.save(os.path.join(save_dir, '%s_rcnns.npy' % (pid)), detections)

            if len(ensembles):
                ensembles = ensembles[:, 1:]
                np.save(os.path.join(save_dir, '%s_ensembles.npy' % (pid)), ensembles)


            # Clear gpu memory
            del b60f_input, b31f_input, truth_bboxes, truth_labels, image#, gt_mask, gt_img, pred_img, full, score
            torch.cuda.empty_cache()

        except Exception as e:
            del b60f_input, b31f_input, truth_bboxes, truth_labels, image,
            torch.cuda.empty_cache()
            traceback.print_exc()
                        
            print
            return
    
    # Generate prediction csv for the use of performning FROC analysis
    # Save both rpn and rcnn results
    rpn_res = []
    rcnn_res = []
    ensemble_res = []
    for pid in dataset.b60_filenames:
        if os.path.exists(os.path.join(save_dir, '%s_rpns.npy' % (pid))):
            rpns = np.load(os.path.join(save_dir, '%s_rpns.npy' % (pid)))
            rpns = rpns[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(rpns))
            rpn_res.append(np.concatenate([names, rpns], axis=1))

        if os.path.exists(os.path.join(save_dir, '%s_rcnns.npy' % (pid))):
            rcnns = np.load(os.path.join(save_dir, '%s_rcnns.npy' % (pid)))
            rcnns = rcnns[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(rcnns))
            rcnn_res.append(np.concatenate([names, rcnns], axis=1))

        if os.path.exists(os.path.join(save_dir, '%s_ensembles.npy' % (pid))):
            ensembles = np.load(os.path.join(save_dir, '%s_ensembles.npy' % (pid)))
            ensembles = ensembles[:, [3, 2, 1, 4, 0]]
            names = np.array([[pid]] * len(ensembles))
            ensemble_res.append(np.concatenate([names, ensembles], axis=1))
    
    rpn_res = np.concatenate(rpn_res, axis=0)
    rcnn_res = np.concatenate(rcnn_res, axis=0)
    ensemble_res = np.concatenate(ensemble_res, axis=0)
    col_names = ['seriesuid','coordX','coordY','coordZ','diameter_mm', 'probability']
    eval_dir = os.path.join(save_dir, 'FROC')
    rpn_submission_path = os.path.join(eval_dir, 'submission_rpn.csv')
    rcnn_submission_path = os.path.join(eval_dir, 'submission_rcnn.csv')
    ensemble_submission_path = os.path.join(eval_dir, 'submission_ensemble.csv')
    
    df = pd.DataFrame(rpn_res, columns=col_names)
    df.to_csv(rpn_submission_path, index=False)

    df = pd.DataFrame(rcnn_res, columns=col_names)
    df.to_csv(rcnn_submission_path, index=False)

    df = pd.DataFrame(ensemble_res, columns=col_names)
    df.to_csv(ensemble_submission_path, index=False)

def noduleCADEval(dataset, save_dir,epoch,out_dir,fold_num,pvalue):
    # Start evaluating
    eval_dir = os.path.join(save_dir, 'FROC')
    rpn_submission_path = os.path.join(eval_dir, 'submission_rpn.csv')
    rcnn_submission_path = os.path.join(eval_dir, 'submission_rcnn.csv')
    ensemble_submission_path = os.path.join(eval_dir, 'submission_ensemble.csv')
    
    if not os.path.exists(os.path.join(eval_dir, 'rpn')):
        os.makedirs(os.path.join(eval_dir, 'rpn'))
    if not os.path.exists(os.path.join(eval_dir, 'rcnn')):
        os.makedirs(os.path.join(eval_dir, 'rcnn'))
    if not os.path.exists(os.path.join(eval_dir, 'ensemble')):
        os.makedirs(os.path.join(eval_dir, 'ensemble'))

    rpn_sens_boot,rpn_froc_boot,rpn_sens_norm,rpn_froc_norm = noduleCADEvaluation('./evaluationScript/annos-fold'+fold_num+'/annotations-final.csv',
    './evaluationScript/annotations_excluded.csv',
    './evaluationScript/annos-fold'+fold_num+'/seriesuids-final.csv', rpn_submission_path, os.path.join(eval_dir, 'rpn'))

    rcnn_sens_boot,rcnn_froc_boot,rcnn_sens_norm,rcnn_froc_norm =noduleCADEvaluation('./evaluationScript/annos-fold'+fold_num+'/annotations-final.csv',
    './evaluationScript/annotations_excluded.csv',
    './evaluationScript/annos-fold'+fold_num+'/seriesuids-final.csv', rcnn_submission_path, os.path.join(eval_dir, 'rcnn'))

    ensemble_sens_boot,ensemble_froc_boot,ensemble_sens_norm,ensemble_froc_norm =noduleCADEvaluation('./evaluationScript/annos-fold'+fold_num+'/annotations-final.csv',
    './evaluationScript/annotations_excluded.csv',
    './evaluationScript/annos-fold'+fold_num+'/seriesuids-final.csv', ensemble_submission_path, os.path.join(eval_dir, 'ensemble'))
    content=[]
    content.append([str(epoch),'froc(avg)','0.125','0.25','0.5','1','2','4','8'])
    content.append(['rpn_boot']+[rpn_froc_boot]+rpn_sens_boot)
    content.append(['rcnn_boot']+[rcnn_froc_boot]+rcnn_sens_boot)
    content.append(['ensemble_boot']+[ensemble_froc_boot]+ensemble_sens_boot)
    content.append(['rpn_norm']+[rpn_froc_norm]+rpn_sens_norm)
    content.append(['rcnn_norm']+[rcnn_froc_norm]+rcnn_sens_norm)
    content.append(['ensemble_norm']+[ensemble_froc_norm]+ensemble_sens_norm)
    content_dataframe = pd.DataFrame(content)
#     types = 'include_fp'
#     if not os.path.exists(out_dir+'froc_'+types+'.csv'):
#         content_dataframe.to_csv(out_dir+'froc_'+types+'.csv',index=False,header=None)
#     else:
#         exist_dataframe = pd.read_csv(out_dir+'froc_'+types+'.csv',header=None)
#         new_dataframe = pd.concat([exist_dataframe,content_dataframe],axis=0)
#         new_dataframe.to_csv(out_dir+'froc_'+types+'.csv',index=False,header=None)
        
    simple = []
    simple.append([str(pvalue),rpn_froc_boot,rcnn_froc_boot,ensemble_froc_boot,rpn_froc_norm,rcnn_froc_norm,ensemble_froc_norm])
    simple_dataframe = pd.DataFrame(simple)
#     types='simple'
#     if not os.path.exists(out_dir+'froc_'+types+'.csv'):
#         simple_dataframe.to_csv(out_dir+'froc_'+types+'.csv',index=False,header=['epoch','rpn_boot','rcnn_boot','ensemble_boot','rpn_norm','rcnn_norm','ensemble_norm'])
#     else:
#         exist_dataframe = pd.read_csv(out_dir+'froc_'+types+'.csv',header=None)
#         new_dataframe = pd.concat([exist_dataframe,simple_dataframe],axis=0)
#         new_dataframe.to_csv(out_dir+'froc_'+types+'.csv',index=False,header=None)
    types='pvalue'
    if not os.path.exists(out_dir+'froc_'+types+'.csv'):
        simple_dataframe.to_csv(out_dir+'froc_'+types+'.csv',index=False,header=['epoch','rpn_boot','rcnn_boot','ensemble_boot','rpn_norm','rcnn_norm','ensemble_norm'])
    else:
        exist_dataframe = pd.read_csv(out_dir+'froc_'+types+'.csv',header=None)
        new_dataframe = pd.concat([exist_dataframe,simple_dataframe],axis=0)
        new_dataframe.to_csv(out_dir+'froc_'+types+'.csv',index=False,header=None)
    print
    print


def eval_single(net, input):
    with torch.no_grad():
        input = input.cuda().unsqueeze(0)
        logits = net.forward(input)
        logits = logits[0]
    
    masks = logits.cpu().data.numpy()
    masks = (masks > 0.5).astype(np.int32)
    return masks
 

if __name__ == '__main__':
    main()
