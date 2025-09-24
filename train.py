from net.nodule_net import NoduleNet
import time
from dataset.collate import train_collate, test_collate, eval_collate
from dataset.bbox_reader import BboxReader
# from dataset.mask_reader import MaskReader
from utils.util import Logger
from config import train_config, data_config, net_config, config
import pprint
from torch.utils.data import DataLoader, ConcatDataset
from torch.autograd import Variable
import torch
import numpy as np
import argparse
import os
import sys
from tqdm import tqdm
import random
import traceback
import shutil
# from torch.utils.tensorboard import SummaryWriter



this_module = sys.modules[__name__]


parser = argparse.ArgumentParser(description='PyTorch Detector')
parser.add_argument('--net', '-m', metavar='NET', default=train_config['net'],
                    help='neural net')
parser.add_argument('--epochs', default=train_config['epochs'], type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=train_config['batch_size'], type=int, metavar='N',
                    help='batch size')
parser.add_argument('--epoch-detect', default=train_config['epoch_detect'], type=int, metavar='NR',
                    help='number of epochs before training detector')
parser.add_argument('--epoch-rcnn', default=train_config['epoch_rcnn'], type=int, metavar='NR',
                    help='number of epochs before training rcnn')
parser.add_argument('--ckpt', default=train_config['initial_checkpoint'], type=str, metavar='CKPT',
                    help='checkpoint to use')
parser.add_argument('--optimizer', default=train_config['optimizer'], type=str, metavar='SPLIT',
                    help='which split set to use')
parser.add_argument('--init-lr', default=train_config['init_lr'], type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=train_config['momentum'], type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', default=train_config['weight_decay'], type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--epoch-save', default=train_config['epoch_save'], type=int, metavar='S',
                    help='save frequency')
parser.add_argument('--out-dir', default=train_config['out_dir'], type=str, metavar='OUT',
                    help='directory to save results of this training')
parser.add_argument('--train-set-list', default=train_config['train_set_list'], nargs='+', type=str,
                    help='train set paths list')
parser.add_argument('--val-set-list', default=train_config['val_set_list'], nargs='+', type=str,
                    help='val set paths list')
parser.add_argument('--data-dir', default=train_config['DATA_DIR'], type=str, metavar='OUT',
                    help='path to load data')
parser.add_argument('--num-workers', default=train_config['num_workers'], type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--fold', default='1', type=str, metavar='N',
                    help='number of fold')
def copy_dir(dir1, dir2):
    dlist = os.listdir(dir1)
    if not os.path.exists(dir2):
        os.mkdir(dir2)
    for f in dlist:
        file1 = os.path.join(dir1, f)  # 源文件
        file2 = os.path.join(dir2, f)  # 目标文件
        if os.path.isfile(file1):
            shutil.copyfile(file1, file2)
        if os.path.isdir(file1):
            copy_dir(file1, file2)

def main():
    # Load training configuration
    args = parser.parse_args()

    net = args.net
    initial_checkpoint = args.ckpt
    out_dir = args.out_dir
    weight_decay = args.weight_decay
    momentum = args.momentum
    optimizer = args.optimizer
    init_lr = args.init_lr
    epochs = args.epochs
    epoch_save = args.epoch_save
    epoch_detect = args.epoch_detect
    epoch_rcnn = args.epoch_rcnn
    batch_size = args.batch_size
#     train_set_list = args.train_set_list
#     val_set_list = args.val_set_list
    train_set_list = {'B60f':'/home/data2/ltz/domain_adaptation/filelist/filelistB60f.npy',
                       'B31f':'/home/data2/ltz/domain_adaptation/filelist/filelistB31f.npy'}
#     'val_set_list': ['split/3_val.csv'],
    val_set_list = {'B60f':'/home/data2/ltz/domain_adaptation/filelist/filelistB60f.npy',
                      'B31f':'/home/data2/ltz/domain_adaptation/filelist/filelistB31f.npy'}
    num_workers = args.num_workers
    lr_schdule = train_config['lr_schedule']
    datadir = args.data_dir
    label_types = config['label_types']

#     train_dataset_list = []
#     val_dataset_list = []
#     for i in range(len(train_set_list)):
#     b31_train_name = train_set_list[0]
#     b60_train_name = train_set_list[1]
    
#     b31_val_name = val_set_list[0]
#     b60_val_name = val_set_list[1]
#     print(train_set_list)
    train_dataset = BboxReader(datadir,train_set_list,config, mode='train')

    val_dataset = BboxReader(datadir,val_set_list,config, mode='val')
            
        
#     train_loader = DataLoader(ConcatDataset(train_dataset_list), batch_size=batch_size, shuffle=True,
#                               num_workers=num_workers, pin_memory=True, collate_fn=train_collate)
#     val_loader = DataLoader(ConcatDataset(val_dataset_list), batch_size=batch_size, shuffle=False,
#                               num_workers=num_workers, pin_memory=True, collate_fn=train_collate)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, collate_fn=train_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True, collate_fn=train_collate)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                               num_workers=num_workers, pin_memory=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                               num_workers=num_workers, pin_memory=True)
    
    
    # Initilize network
    net = getattr(this_module, net)(net_config)
    net = net.cuda()
    
    optimizer = getattr(torch.optim, optimizer)
    # optimizer = optimizer(net.parameters(), lr=init_lr, weight_decay=weight_decay)
    optimizer = optimizer(net.parameters(), lr=init_lr, weight_decay=weight_decay, momentum=momentum)

    start_epoch = 0

    if initial_checkpoint:
        print('[Loading model from %s]' % initial_checkpoint)
        checkpoint = torch.load(initial_checkpoint)
        start_epoch = checkpoint['epoch']
        state = net.state_dict()
        state.update(checkpoint['state_dict'])

        try:
            net.load_state_dict(state)
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print('Load something failed!')
            traceback.print_exc()

    start_epoch = start_epoch + 1

    model_out_dir = os.path.join(out_dir, 'model')
    tb_out_dir = os.path.join(out_dir, 'runs')
    if not os.path.exists(model_out_dir):
        os.makedirs(model_out_dir)
    logfile = os.path.join(out_dir, 'log_train')
    sys.stdout = Logger(logfile)
    pyfiles = [f for f in os.listdir('./') if f.endswith('.py') or f.endswith('.sh')]
    for f in pyfiles:
        shutil.copy(f,os.path.join(out_dir,f))
    copy_dir('./net/',os.path.join(out_dir,'./net/'))

    print('[Training configuration]')
    for arg in vars(args):
        print(arg, getattr(args, arg))

    print('[Model configuration]')
    pprint.pprint(net_config)

    print('[start_epoch %d, out_dir %s]' % (start_epoch, out_dir))
#     print('[length of train loader %d, length of valid loader %d]' % (len(train_loader), len(val_loader)))
    print('[length of train loader %d]' % (len(train_loader)))
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')

    # Write graph to tensorboard for visualization
#     writer = SummaryWriter(tb_out_dir)
#     train_writer = SummaryWriter(os.path.join(tb_out_dir, 'train'))
#     val_writer = SummaryWriter(os.path.join(tb_out_dir, 'val'))
    # writer.add_graph(net, (torch.zeros((16, 1, 128, 128, 128)).cuda(), [[]], [[]], [[]], [torch.zeros((16, 128, 128, 128))]), verbose=False)
#     lr_scheduler =torch.optim.lr_scheduler.StepLR(optimizer, 50, 0.5)
    for i in tqdm(range(start_epoch, epochs + 1), desc='Total'):
        # learning rate schedule
        if isinstance(optimizer, torch.optim.SGD):
            
            lr = lr_schdule(i, init_lr=init_lr, total=epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = np.inf
        if i >= epoch_detect:
            net.use_detect = True
        else:
            net.use_detect = False
        if i >= epoch_rcnn:
            net.use_rcnn = True
        else:
            net.use_rcnn = False
        print('[epoch %d, lr %f, use_rcnn: %r]' % (i, optimizer.param_groups[0]['lr'], net.use_rcnn))
#         train(net, train_loader, optimizer, i, train_writer)
        train(net, train_loader, optimizer, i)
#         lr_scheduler.step()
        validate(net, val_loader, i)
        print

        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        if i % epoch_save == 0 or i>=epochs-10:
            torch.save({
                'epoch': i,
                'out_dir': out_dir,
                'state_dict': state_dict,
                'optimizer' : optimizer.state_dict()},
                os.path.join(model_out_dir, '%03d.ckpt' % i))

#     writer.close()
#     train_writer.close()
#     val_writer.close()train_set_list


# def train(net, train_loader, optimizer, epoch, writer):
def train(net, train_loader, optimizer, epoch):
    net.set_mode('train')
    s = time.time()
    sr_loss =[]
    rpn_cls_loss, rpn_reg_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []
    total_loss = []
    rpn_stats = []
    rcnn_stats = []

    for j, (b31_input, b60_input, truth_box, truth_label) in tqdm(enumerate(train_loader), total=len(train_loader), desc='Train %d' % epoch):
#         import pdb;pdb.set_trace()
        b31_input = Variable(b31_input).cuda()
        b60_input = Variable(b60_input).cuda()
#         truth_box = np.array(truth_box)
#         truth_label = np.array(truth_label)

        net(b31_input,b60_input, truth_box, truth_label)

        loss, rpn_stat, rcnn_stat = net.loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        sr_loss.append(net.sr_loss.cpu().data.item())
        rpn_cls_loss.append(net.rpn_cls_loss.cpu().data.item())
        rpn_reg_loss.append(net.rpn_reg_loss.cpu().data.item())
        rcnn_cls_loss.append(net.rcnn_cls_loss.cpu().data.item())
        rcnn_reg_loss.append(net.rcnn_reg_loss.cpu().data.item())
        total_loss.append(loss.cpu().data.item())
        rpn_stats.append(rpn_stat)
        rcnn_stats.append(rcnn_stat)
        del b31_input, b60_input, truth_box, truth_label
        del net.total_loss, net.sr_loss, net.rpn_cls_loss, net.rpn_reg_loss, net.rcnn_cls_loss, net.rcnn_reg_loss
        if net.use_detect:
            del net.rpn_proposals, net.detections
            del net.rpn_logits_flat, net.rpn_deltas_flat
        if net.use_rcnn:
            del net.rcnn_logits, net.rcnn_deltas

#     torch.cuda.empty_cache()
    
    
    
    print('Train Epoch %d, iter %d, total time %f, loss %f' % (epoch, j, time.time() - s, np.average(total_loss)))
    print('sr_cls %f, rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f' % \
        (np.average(sr_loss), np.average(rpn_cls_loss), np.average(rpn_reg_loss),
            np.average(rcnn_cls_loss), np.average(rcnn_reg_loss)))
    if net.use_detect:
        rpn_stats = np.asarray(rpn_stats, np.float32)
        print('rpn_stats: tpr %f, tnr %f, total pos %d, total neg %d, reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            100.0 * np.sum(rpn_stats[:, 0]) / np.sum(rpn_stats[:, 1]),
            100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3]),
            np.sum(rpn_stats[:, 1]),
            np.sum(rpn_stats[:, 3]),
            np.mean(rpn_stats[:, 4]),
            np.mean(rpn_stats[:, 5]),
            np.mean(rpn_stats[:, 6]),
            np.mean(rpn_stats[:, 7]),
            np.mean(rpn_stats[:, 8]),
            np.mean(rpn_stats[:, 9])))

    # Write to tensorboard
#     writer.add_scalar('loss', np.average(total_loss), epoch)
#     writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
#     writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
#     writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
#     writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)
#     writer.add_scalar('mask_loss', np.average(mask_loss), epoch)

#     writer.add_scalar('rpn_reg_z', np.mean(rpn_stats[:, 4]), epoch)
#     writer.add_scalar('rpn_reg_y', np.mean(rpn_stats[:, 5]), epoch)
#     writer.add_scalar('rpn_reg_x', np.mean(rpn_stats[:, 6]), epoch)
#     writer.add_scalar('rpn_reg_d', np.mean(rpn_stats[:, 7]), epoch)
#     writer.add_scalar('rpn_reg_h', np.mean(rpn_stats[:, 8]), epoch)
#     writer.add_scalar('rpn_reg_w', np.mean(rpn_stats[:, 9]), epoch)

    if net.use_rcnn:
        confusion_matrix = np.asarray([stat[-1] for stat in rcnn_stats], np.int32)
        rcnn_stats = np.asarray([stat[:-1] for stat in rcnn_stats], np.float32)
        
        confusion_matrix = np.sum(confusion_matrix, 0)

        print('rcnn_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(rcnn_stats[:, 0]),
            np.mean(rcnn_stats[:, 1]),
            np.mean(rcnn_stats[:, 2]),
            np.mean(rcnn_stats[:, 3]),
            np.mean(rcnn_stats[:, 4]),
            np.mean(rcnn_stats[:, 5])))
        # print_confusion_matrix(confusion_matrix)
#         writer.add_scalar('rcnn_reg_z', np.mean(rcnn_stats[:, 0]), epoch)
#         writer.add_scalar('rcnn_reg_y', np.mean(rcnn_stats[:, 1]), epoch)
#         writer.add_scalar('rcnn_reg_x', np.mean(rcnn_stats[:, 2]), epoch)
#         writer.add_scalar('rcnn_reg_d', np.mean(rcnn_stats[:, 3]), epoch)
#         writer.add_scalar('rcnn_reg_h', np.mean(rcnn_stats[:, 4]), epoch)
#         writer.add_scalar('rcnn_reg_w', np.mean(rcnn_stats[:, 5]), epoch)
    
    print
    print
    

def validate(net, val_loader, epoch):
    net.set_mode('valid')
    sr_loss=[]
    rpn_cls_loss, rpn_reg_loss = [], []
    rcnn_cls_loss, rcnn_reg_loss = [], []
    total_loss = []
    rpn_stats = []
    rcnn_stats = []

    s = time.time()
    for j, (b31_input, b60_input, truth_box, truth_label) in tqdm(enumerate(val_loader), total=len(val_loader), desc='Train %d' % epoch):
        with torch.no_grad():
            b31_input = Variable(b31_input).cuda()
            b60_input = Variable(b60_input).cuda()
#             truth_box = np.array(truth_box)
#             truth_label = np.array(truth_label)

            net(b31_input,b60_input, truth_box, truth_label)
            loss, rpn_stat, rcnn_stat = net.loss()
        sr_loss.append(net.sr_loss.cpu().data.item())
        rpn_cls_loss.append(net.rpn_cls_loss.cpu().data.item())
        rpn_reg_loss.append(net.rpn_reg_loss.cpu().data.item())
        rcnn_cls_loss.append(net.rcnn_cls_loss.cpu().data.item())
        rcnn_reg_loss.append(net.rcnn_reg_loss.cpu().data.item())
        
        total_loss.append(loss.cpu().data.item())
        rpn_stats.append(rpn_stat)
        rcnn_stats.append(rcnn_stat)


    rpn_stats = np.asarray(rpn_stats, np.float32)
    print('Val Epoch %d, iter %d, total time %f, loss %f' % (epoch, j, time.time()-s, np.average(total_loss)))
    print('sr_cls %f, rpn_cls %f, rpn_reg %f, rcnn_cls %f, rcnn_reg %f' % \
        (np.average(sr_loss), np.average(rpn_cls_loss), np.average(rpn_reg_loss),
            np.average(rcnn_cls_loss), np.average(rcnn_reg_loss)))
    if net.use_detect:
        print('rpn_stats: tpr %f, tnr %f, total pos %d, total neg %d, reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            100.0 * np.sum(rpn_stats[:, 0]) / np.sum(rpn_stats[:, 1]),
            100.0 * np.sum(rpn_stats[:, 2]) / np.sum(rpn_stats[:, 3]),
            np.sum(rpn_stats[:, 1]),
            np.sum(rpn_stats[:, 3]),
            np.mean(rpn_stats[:, 4]),
            np.mean(rpn_stats[:, 5]),
            np.mean(rpn_stats[:, 6]),
            np.mean(rpn_stats[:, 7]),
            np.mean(rpn_stats[:, 8]),
            np.mean(rpn_stats[:, 9])))
    
    # Write to tensorboard
#     writer.add_scalar('loss', np.average(total_loss), epoch)
#     writer.add_scalar('rpn_cls', np.average(rpn_cls_loss), epoch)
#     writer.add_scalar('rpn_reg', np.average(rpn_reg_loss), epoch)
#     writer.add_scalar('rcnn_cls', np.average(rcnn_cls_loss), epoch)
#     writer.add_scalar('rcnn_reg', np.average(rcnn_reg_loss), epoch)

#     writer.add_scalar('rpn_reg_z', np.mean(rpn_stats[:, 4]), epoch)
#     writer.add_scalar('rpn_reg_y', np.mean(rpn_stats[:, 5]), epoch)
#     writer.add_scalar('rpn_reg_x', np.mean(rpn_stats[:, 6]), epoch)
#     writer.add_scalar('rpn_reg_d', np.mean(rpn_stats[:, 7]), epoch)
#     writer.add_scalar('rpn_reg_h', np.mean(rpn_stats[:, 8]), epoch)
#     writer.add_scalar('rpn_reg_w', np.mean(rpn_stats[:, 9]), epoch)

    if net.use_rcnn:
        confusion_matrix = np.asarray([stat[-1] for stat in rcnn_stats], np.int32)
        rcnn_stats = np.asarray([stat[:-1] for stat in rcnn_stats], np.float32)
        
        confusion_matrix = np.sum(confusion_matrix, 0)
        print('rcnn_stats: reg %.4f, %.4f, %.4f, %.4f, %.4f, %.4f' % (
            np.mean(rcnn_stats[:, 0]),
            np.mean(rcnn_stats[:, 1]),
            np.mean(rcnn_stats[:, 2]),
            np.mean(rcnn_stats[:, 3]),
            np.mean(rcnn_stats[:, 4]),
            np.mean(rcnn_stats[:, 5])))
        # print_confusion_matrix(confusion_matrix)
#         writer.add_scalar('rcnn_reg_z', np.mean(rcnn_stats[:, 0]), epoch)
#         writer.add_scalar('rcnn_reg_y', np.mean(rcnn_stats[:, 1]), epoch)
#         writer.add_scalar('rcnn_reg_x', np.mean(rcnn_stats[:, 2]), epoch)
#         writer.add_scalar('rcnn_reg_d', np.mean(rcnn_stats[:, 3]), epoch)
#         writer.add_scalar('rcnn_reg_h', np.mean(rcnn_stats[:, 4]), epoch)
#         writer.add_scalar('rcnn_reg_w', np.mean(rcnn_stats[:, 5]), epoch)

    print
    print
    
    del b31_input, b60_input, truth_box, truth_label
    del net.total_loss, net.rpn_cls_loss, net.rpn_reg_loss, net.rcnn_cls_loss, net.rcnn_reg_loss
    if net.use_detect:
        del net.rpn_proposals, net.detections
        del net.rpn_logits_flat, net.rpn_deltas_flat
    if net.use_rcnn:
        del net.rcnn_logits, net.rcnn_deltas

    torch.cuda.empty_cache()

def print_confusion_matrix(confusion_matrix):
    line_new = '{:>4}  ' * (len(config['roi_names']) + 2)
    print(line_new.format('gt/p', *list(range(len(config['roi_names']) + 1))))

    for i in range(len(config['roi_names']) + 1):
        print(line_new.format(i, *list(confusion_matrix[i])))
        

if __name__ == '__main__':
    main()



