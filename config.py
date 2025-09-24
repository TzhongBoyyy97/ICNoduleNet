import os
import numpy as np
import torch
import random


# Set seed
SEED = 35202
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Preprocessing using preserved HU in dilated part of mask
BASE = '/home/htang6/workspace/data/LIDC/' # make sure you have the ending '/'
data_config = {
    # put combined LUNA16 .mhd files into one folder
    'data_dir': BASE + 'combined',

    # directory for putting all preprocessed results for training to this path
    'preprocessed_data_dir': '/root/workspace/imgctl/preprocess_result_BCB31f/',

    # put annotation downloaded from LIDC to this path
    'annos_dir': BASE + 'annotation/LIDC-XML-only/tcia-lidc-xml',

    # put lung mask downloaded from LUNA16 to this path
    'lung_mask_dir': BASE + 'seg-lungs-LUNA16/',

    # Directory for saving intermediate results
    'ctr_arr_save_dir': BASE + 'annotation/mask_test',
    'mask_save_dir': BASE + 'masks_test',
    'mask_exclude_save_dir': BASE + 'masks_exclude_test',


    'roi_names': ['nodule'],
    'crop_size': [128, 128, 128],
    'bbox_border': 8,
    'pad_value': 170,
    # 'jitter_range': [0, 0, 0],
}


def get_anchors(bases, aspect_ratios):
    anchors = []
    for b in bases:
        for asp in aspect_ratios:
#             np.random.normal(loc=0.0, scale=1.0, size=None)
            d, h, w = b * asp[0], b * asp[1], b * asp[2]
#             d, h, w = np.random.normal(loc=b, scale=1.0, size=None) * asp[0], np.random.normal(loc=b, scale=1.0, size=None) * asp[1], np.random.normal(loc=b, scale=1.0, size=None) * asp[2]
            anchors.append([d, h, w])

    return anchors

bases = [5, 10, 20, 30, 50]
# bases = [5, 20, 50]
aspect_ratios = [[1, 1, 1]]

net_config = {
    # Net configuration
    'anchors': get_anchors(bases, aspect_ratios),
    'chanel': 1,
    'crop_size': data_config['crop_size'],
    'stride': 4,
    'max_stride': 16,
    'num_neg': 800,
    'th_neg': 0.02,
    'th_pos_train': 0.5,
    'th_pos_val': 1,
    'num_hard': 3,
    'bound_size': 12,
    'blacklist': [],

    'augtype': {'flip': True, 'rotate': True, 'scale': True, 'swap': False},
    'r_rand_crop': 0.,
    'pad_value': 170,
    
    #sr block configuration
    'srchannel':16,
    # region proposal network configuration
    'rpn_train_bg_thresh_high': 0.02,
    'rpn_train_fg_thresh_low': 0.5,
    
    'rpn_train_nms_num': 300,
    'rpn_train_nms_pre_score_threshold': 0.5,
    'rpn_train_nms_overlap_threshold': 0.1,
    'rpn_test_nms_pre_score_threshold': 0.5,
    'rpn_test_nms_overlap_threshold': 0.1,

    # false positive reduction network configuration
    'num_class': len(data_config['roi_names']) + 1,
    'rcnn_crop_size': (7,7,7), # can be set smaller, should not affect much
    'rcnn_train_fg_thresh_low': 0.5,
    'rcnn_train_bg_thresh_high': 0.1,
    'rcnn_train_batch_size': 32,
    'rcnn_train_fg_fraction': 0.5,
    'rcnn_train_nms_pre_score_threshold': 0.5,
    'rcnn_train_nms_overlap_threshold': 0.1,
    'rcnn_test_nms_pre_score_threshold': 0.0,
    'rcnn_test_nms_overlap_threshold': 0.1,

    'mask_crop_size': [24, 48, 48],
    'mask_test_nms_overlap_threshold': 0.3,

    'box_reg_weight': [1., 1., 1., 1., 1., 1.]
    
    
}


def lr_shedule(epoch, init_lr=0.01, total=500):
    if epoch <= total * 0.5:
        lr = init_lr
    elif epoch <= total * 0.7:
        lr = 0.1 * init_lr
    else:
        lr = 0.01 * init_lr
    return lr

train_config = {
    'net': 'NoduleNet',
    'batch_size': 16,
    'lr_schedule': lr_shedule,
    'optimizer': 'SGD',
    'momentum': 0.9,
    'weight_decay': 1e-4,

    'epochs': 300,
    'epoch_save': 10,
    'epoch_detect':11,
    'epoch_rcnn': 21,
    'epoch_mask': 10000,
    'num_workers': 8,

#     'train_set_list': ['/root/workspace/imgctl/filelists/5_cross_val_filesets_seed2/filelistB31f_train_fold5.npy'],
    'train_set_list': {'B60f':'/home/data2/ltz/domain_adaptation/filelist/filelistB60f.npy',
                       'B31f':'/home/data2/ltz/domain_adaptation/filelist/filelistB31f.npy'},
#     'val_set_list': ['split/3_val.csv'],
    'val_set_list': {'B60f':'/home/data2/ltz/domain_adaptation/filelist/filelistB60f.npyy',
                      'B31f':'/home/data2/ltz/domain_adaptation/filelist/filelistB31f.npy'},
    'test_set_name': '/root/autodl-tmp/des_filesets/filelistB60f_test_fold5.npy',
    'label_types': ['bbox'],
    'DATA_DIR': {'B60f':'/home/data2/ltz/domain_adaptation/data/preprocess_result_BCB60f/','B31f':'/home/data2/ltz/domain_adaptation/data/preprocess_result_BCB31f/'},
    'ROOT_DIR': os.getcwd()
}

if train_config['optimizer'] == 'SGD':
    train_config['init_lr'] = 0.01
elif train_config['optimizer'] == 'Adam':
    train_config['init_lr'] = 0.001
elif train_config['optimizer'] == 'RMSprop':
    train_config['init_lr'] = 2e-3


train_config['RESULTS_DIR'] = os.path.join(train_config['ROOT_DIR'], 'results')
train_config['out_dir'] = os.path.join(train_config['RESULTS_DIR'], 'fold5')
train_config['initial_checkpoint'] = None 
# train_config['initial_checkpoint'] = './new_pretrain.ckpt'


config = dict(data_config, **net_config)
config = dict(config, **train_config)
