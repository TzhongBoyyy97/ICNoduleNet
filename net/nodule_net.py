import sys

from net.layer import *

from config import net_config as config
import copy
from torch.nn.parallel.data_parallel import data_parallel
import time
import torch.nn.functional as F
from utils.util import center_box_to_coord_box, ext2factor, clip_boxes
from torch.nn.parallel import data_parallel
import random
from scipy.stats import norm


bn_momentum = 0.1
affine = True

class ResBlock3d(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm3d(n_out, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm3d(n_out, momentum=bn_momentum)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size = 1, stride = stride),
                nn.BatchNorm3d(n_out, momentum=bn_momentum))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class FeatureNet(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super(FeatureNet, self).__init__()
        self.preBlock = nn.Sequential(
            nn.Conv3d(in_channels, 24, kernel_size = 3, padding = 1, stride=2),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24, momentum=bn_momentum),
            nn.ReLU(inplace = True))

        self.forw1 = nn.Sequential(
            ResBlock3d(24, 32),
            ResBlock3d(32, 32))

        self.forw2 = nn.Sequential(
            ResBlock3d(32, 64),
            ResBlock3d(64, 64))

        self.forw3 = nn.Sequential(
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64))

        self.forw4 = nn.Sequential(
            ResBlock3d(64, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64))

        # skip connection in U-net
        self.back2 = nn.Sequential(
            # 64 + 64 + 3, where 3 is the channeld dimension of coord
            ResBlock3d(128, 128),
            ResBlock3d(128, 128),
            ResBlock3d(128, 128))

        # skip connection in U-net
        self.back3 = nn.Sequential(
            ResBlock3d(128, 64),
            ResBlock3d(64, 64),
            ResBlock3d(64, 64))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2,
                                     return_indices=True)

        # upsampling in U-net
        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))

        # upsampling in U-net
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


    def forward(self, x):
        out = self.preBlock(x)#16
        out_pool = out
        out1 = self.forw1(out_pool)#32
        out1_pool, _ = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)#64
        #out2 = self.drop(out2)
        out2_pool, _ = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)#96
        out3_pool, _ = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)#96
        #out4 = self.drop(out4)

        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))#96+96
        rev2 = self.path2(comb3)
        comb2 = self.back2(torch.cat((rev2, out2), 1))#64+64

        return [x, out1, comb2], out2

class SELayer(nn.Module):
    def __init__(self, channel, reduction=32):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Sequential(
                nn.Conv3d(channel, reduction, kernel_size=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(reduction, channel, kernel_size=1, bias=False),
                nn.Sigmoid()
        )
#         self.fc = nn.Sequential(
#                 nn.Linear(channel, channel // reduction),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(channel // reduction, channel),
#                 nn.Sigmoid()
#         )
    def forward(self, x):
        #         print(x.shape)
        out = self.avg_pool(x)
#         print(out.shape)
        out = self.conv(out)
        output=out*x
        return output
        
        
    
class ICBlock(nn.Module):
    def __init__(self,cfg):
        super(ICBlock, self).__init__()
#         self.rblock = nn.Sequential(
#             nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
#         )
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.convTran2 = nn.Sequential(
            nn.ConvTranspose3d(1, 1, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
#         self.input = nn.Conv3d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
#         self.conv1 = nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.se = SELayer(channel = 128, reduction = 64)
        
    def forward(self, comb2):
#         out = self.input1(comb2)
#         out = self.relu(out)
#         out = self.convTran1(out)
#         out = self.se(out)
#         out = self.input2(out)
#         out = self.convTran2(out)
#         out = self.rblock(comb2)#128*32*32
#         out = self.conv1(out)
        out = self.se(comb2)
        out = self.upsample(out)
        out = self.conv2(out)
        out = self.convTran2(out)
        return out
    
class RpnHead(nn.Module):
    def __init__(self, config, in_channels=128):
        super(RpnHead, self).__init__()
        self.drop = nn.Dropout3d(p=0.5, inplace=False)
        self.conv = nn.Sequential(nn.Conv3d(in_channels, 64, kernel_size=1),
                                    nn.ReLU())
        self.logits = nn.Conv3d(64, 1 * len(config['anchors']), kernel_size=1)
        self.deltas = nn.Conv3d(64, 6 * len(config['anchors']), kernel_size=1)

    def forward(self, f):
        # out = self.drop(f)
        out = self.conv(f)

        logits = self.logits(out)
        deltas = self.deltas(out)
        size = logits.size()
        logits = logits.view(logits.size(0), logits.size(1), -1)
        logits = logits.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 1)
        
        size = deltas.size()
        deltas = deltas.view(deltas.size(0), deltas.size(1), -1)
        deltas = deltas.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 6)
        

        return logits, deltas

class RcnnHead(nn.Module):
    def __init__(self, cfg, in_channels=128):
        super(RcnnHead, self).__init__()
        self.num_class = cfg['num_class']
        self.crop_size = cfg['rcnn_crop_size']

        self.fc1 = nn.Linear(in_channels * self.crop_size[0] * self.crop_size[1] * self.crop_size[2], 512)
        self.fc2 = nn.Linear(512, 256)
        self.logit = nn.Linear(256, self.num_class)
        self.delta = nn.Linear(256, self.num_class * 6)

    def forward(self, crops):
        x = crops.view(crops.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        # x = F.dropout(x, 0.5, training=self.training)
        logits = self.logit(x)
        deltas = self.delta(x)

        return logits, deltas




def top1pred(boxes):
    res = []
    pred_cats = np.unique(boxes[:, -1])
    for cat in pred_cats:
        preds = boxes[boxes[:, -1] == cat]
        res.append(preds[0])
        
    res = np.array(res)
    return res

def random1pred(boxes):
    res = []
    pred_cats = np.unique(boxes[:, -1])
    for cat in pred_cats:
        preds = boxes[boxes[:, -1] == cat]
        idx = random.sample(range(len(preds)), 1)[0]
        res.append(preds[idx])
        
    res = np.array(res)
    return res


class CropRoi(nn.Module):
    def __init__(self, cfg, rcnn_crop_size):
        super(CropRoi, self).__init__()
        self.cfg = cfg
        self.rcnn_crop_size  = rcnn_crop_size
        self.scale = cfg['stride']
        self.DEPTH, self.HEIGHT, self.WIDTH = cfg['crop_size'] 

    def forward(self, f, inputs, proposals):
        self.DEPTH, self.HEIGHT, self.WIDTH = inputs.shape[2:]

        crops = []
        for p in proposals:
            b = int(p[0])
            center = p[2:5]
            side_length = p[5:8]
            c0 = center - side_length / 2 # left bottom corner
            c1 = c0 + side_length # right upper corner
            c0 = (c0 / self.scale).floor().long()
            c1 = (c1 / self.scale).ceil().long()
            minimum = torch.LongTensor([[0, 0, 0]]).cuda()
            maximum = torch.LongTensor(
                np.array([[self.DEPTH, self.HEIGHT, self.WIDTH]]) / self.scale).cuda()

            c0 = torch.cat((c0.unsqueeze(0), minimum), 0)
            c1 = torch.cat((c1.unsqueeze(0), maximum), 0)
            c0, _ = torch.max(c0, 0)
            c1, _ = torch.min(c1, 0)

            # Slice 0 dim, should never happen
            if np.any((c1 - c0).cpu().data.numpy() < 1):
                print(p)
                print('c0:', c0, ', c1:', c1)
            crop = f[b, :, c0[0]:c1[0], c0[1]:c1[1], c0[2]:c1[2]]
            crop = F.adaptive_max_pool3d(crop, self.rcnn_crop_size)
            crops.append(crop)

        crops = torch.stack(crops)

        return crops

class NoduleNet(nn.Module):
    def __init__(self, cfg, mode='train'):
        super(NoduleNet, self).__init__()

        self.cfg = cfg
        self.mode = mode
        self.feature_net = FeatureNet(config, 1, 128)
        self.rpn = RpnHead(cfg, in_channels=128)
        self.rcnn_head = RcnnHead(cfg, in_channels=64)
        self.rcnn_crop = CropRoi(cfg, cfg['rcnn_crop_size'])
        self.icblock_layer = ICBlock(cfg)
        self.use_detect = False
        self.use_rcnn = False

        # self.rpn_loss = Loss(cfg['num_hard'])
        

    def forward(self, b31_inputs, b60_inputs, truth_boxes, truth_labels, split_combiner=None, nzhw=None):
        features, feat_4 = data_parallel(self.feature_net, (b60_inputs)); #print('fs[-1] ', fs[-1].shape)
        fs = features[-1]
        self.b31_inputs = b31_inputs
        self.ic_out = self.icblock_layer(fs)
        self.generate_patch = b60_inputs + self.ic_out
        assert(b31_inputs.shape == self.generate_patch.shape)
        
        if self.use_detect:
            self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn, fs)

            b,D,H,W,_,num_class = self.rpn_logits_flat.shape

            self.rpn_logits_flat = self.rpn_logits_flat.view(b, -1, 1);#print('rpn_logit ', self.rpn_logits_flat.shape)
            self.rpn_deltas_flat = self.rpn_deltas_flat.view(b, -1, 6);#print('rpn_delta ', self.rpn_deltas_flat.shape)


            self.rpn_window    = make_rpn_windows(fs, self.cfg, self.mode)
            self.rpn_proposals = []
            if self.use_rcnn or self.mode in ['eval', 'test']:
                self.rpn_proposals = rpn_nms(self.cfg, self.mode, self.generate_patch, self.rpn_window,
                      self.rpn_logits_flat, self.rpn_deltas_flat)
                # print 'length of rpn proposals', self.rpn_proposals.shape

            if self.mode in ['train', 'valid']:
                # self.rpn_proposals = torch.zeros((0, 8)).cuda()
                self.rpn_labels, self.rpn_label_assigns, self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights = \
                    make_rpn_target(self.cfg, self.mode, self.generate_patch, self.rpn_window, truth_boxes, truth_labels )

                if self.use_rcnn:
                    # self.rpn_proposals = torch.zeros((0, 8)).cuda()
                    self.rpn_proposals, self.rcnn_labels, self.rcnn_assigns, self.rcnn_targets = \
                        make_rcnn_target(self.cfg, self.mode, self.generate_patch, self.rpn_proposals,
                            truth_boxes, truth_labels)

            #rcnn proposals
            self.detections = copy.deepcopy(self.rpn_proposals)
            self.ensemble_proposals = copy.deepcopy(self.rpn_proposals)

            if self.use_rcnn:
                if len(self.rpn_proposals) > 0:
                    rcnn_crops = self.rcnn_crop(feat_4, self.generate_patch, self.rpn_proposals)
                    self.rcnn_logits, self.rcnn_deltas = data_parallel(self.rcnn_head, rcnn_crops)
                    self.detections, self.keeps = rcnn_nms(self.cfg, self.mode, self.generate_patch, self.rpn_proposals, 
                                                                            self.rcnn_logits, self.rcnn_deltas)

                if self.mode in ['eval']:
                    # Ensemble
                    fpr_res = get_probability(self.cfg, self.mode, self.generate_patch, self.rpn_proposals,  self.rcnn_logits, self.rcnn_deltas)
                    self.ensemble_proposals[:, 1] = (self.ensemble_proposals[:, 1] + fpr_res[:, 0]) / 2


    def loss(self, targets=None):
        cfg  = self.cfg
        self.rpn_cls_loss, self.rpn_reg_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        self.rcnn_cls_loss, self.rcnn_reg_loss = torch.zeros(1).cuda(), torch.zeros(1).cuda()
        rpn_stats = None
        rcnn_stats = None

        self.sr_loss = torch.zeros(1).cuda()
#         self.square_error = nn.MSELoss()
        self.square_error = L1_Charbonnier_loss().cuda()
        self.sr_loss = self.square_error(self.generate_patch, self.b31_inputs)
        if self.use_detect:
            self.rpn_cls_loss, self.rpn_reg_loss, rpn_stats = \
               rpn_loss( self.rpn_logits_flat, self.rpn_deltas_flat, self.rpn_labels,
                self.rpn_label_weights, self.rpn_targets, self.rpn_target_weights, self.cfg, mode=self.mode)
    
        if self.use_rcnn:
            self.rcnn_cls_loss, self.rcnn_reg_loss, rcnn_stats = \
                rcnn_loss(self.rcnn_logits, self.rcnn_deltas, self.rcnn_labels, self.rcnn_targets)

    
        self.total_loss = self.rpn_cls_loss + self.rpn_reg_loss \
                          + self.rcnn_cls_loss +  self.rcnn_reg_loss \
                          + 1*self.sr_loss

    
        return self.total_loss, rpn_stats, rcnn_stats

    def set_mode(self, mode):
        assert mode in ['train', 'valid', 'eval', 'test']
        self.mode = mode
        if mode in ['train']:
            self.train()
        else:
            self.eval()

class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss. From PyTorch LapSRN"""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self,predict,target):
        return torch.mean(torch.sqrt(torch.pow((predict-target), 2) + self.eps)) # epsilon=1e-3

if __name__ == '__main__':
    net = FasterRcnn(config)

    input = torch.rand([4,1,128,128,128])
    input = Variable(input)

