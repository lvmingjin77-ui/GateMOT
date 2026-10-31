from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from loguru import logger as lg

def boolean(value):
  if isinstance(value, bool):
    return value
  if value.lower() in ('true', '1', 'yes'):
    return True
  if value.lower() in ('false', '0', 'no'):
    return False
  raise argparse.ArgumentTypeError('Expected true or false')


class opts(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser()
    # task 
    self.parser.add_argument('--task', default='tracking.ctdet',
                             help='ctdet | ddd | multi_pose '
                             '| tracking or combined with ,')
    self.parser.add_argument('--exp_id', default='mot_exp')
    self.parser.add_argument('--debug', type=int, default= -1,
                             help='level of visualization.'
                                  '1: only show the final detection results'
                                  '2: show the network output features'
                                  '3: use matplot to display' # debug = -1 for training and test or demo use 1-4
                                  '4: save all visualizations to disk')
    self.parser.add_argument('--demo', default='', 
                             help='path to image/ image folders/ video. '
                                  'or "webcam"')
    self.parser.add_argument('--resume', action='store_true',
                             help='resume model and optimizer state from --load_model')
    self.parser.add_argument('--dataset', default='custom', choices=['custom'],
                             help='MOT-format data loaded from the specified image directory and annotation JSON')
    
    # system  
    self.parser.add_argument('--seed', type=int, default=317, 
                             help='random seed') # from CornerNet
    
    # DDP mode
    self.parser.add_argument('--local_rank', type=int, default = -1 )
    self.parser.add_argument('--local-rank', type=int, default = -1, dest='local_rank' )
    
    # visual
    self.parser.add_argument('--save_imgs', default='', help='')
    self.parser.add_argument('--save_img_suffix', default='', help='')
    self.parser.add_argument('--save_video', action='store_true')# test
    self.parser.add_argument('--save_framerate', type=int, default= 30 )
    self.parser.add_argument('--video_h', type=int, default=544, help='')
    self.parser.add_argument('--video_w', type=int, default=1024, help='')
    self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                             help='visualization threshold.')
    self.parser.add_argument('--debugger_theme', default='white', 
                             choices=['white', 'black'])
    self.parser.add_argument('--show_track_color', action='store_true')# F -- use for vis
    self.parser.add_argument('--not_show_bbox', action='store_true')# F --- use for debugger/vis
    self.parser.add_argument('--not_show_number', action='store_true')# F  --- use for debugger/vis
    self.parser.add_argument('--qualitative', action='store_true')# ---- use for debugger / vis the size of font
    self.parser.add_argument('--tango_color', action='store_true')# ---- use for vis / test different color of bbox for vis
    self.parser.add_argument('--only_show_dots', action='store_true')# -----use for vis -- show the object as points instead of bbox
    self.parser.add_argument('--show_trace', action='store_true')# -----use for vis -- show trajectories

    # model sets
    self.parser.add_argument('--use_qattn', action='store_true',
                             help='use GateMOT Q-Attention heads; omitted selects the Conv baseline')
    self.parser.add_argument('--load_model', default='',
                             help='path to pretrained model')
    self.parser.add_argument('--arch', default='dla34', 
                             help='DLA backbone: dla34 for BEE24/MOT17/MOT20, dla60 for SportsMOT')
    self.parser.add_argument('--dla_node', default='dcn') 
    self.parser.add_argument('--no_pretrained', action='store_true', help='Random initialization for smoke tests, not reported training')
    self.parser.add_argument('--allow_partial_load', action='store_true', help='Explicit transfer initialization only; not benchmark evaluation')
    self.parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    self.parser.add_argument('--weight_decay', type=float, default=0.0)
    self.parser.add_argument('--warmup_epochs', type=int, default=0)
    self.parser.add_argument('--output_dir', default='')
    self.parser.add_argument('--fuse', action='store_true') 
    self.parser.add_argument('--head_conv', type=int, default= 256,
                             help='hidden channels in each prediction head; '
                                  '-1 selects the default width.')
    self.parser.add_argument('--num_head_conv', type=int, default=1)
    self.parser.add_argument('--head_kernel', type=int, default=3,
                             help='first Conv baseline layer kernel size; Q-Attention uses 1x1 projections')
    self.parser.add_argument('--down_ratio', type=int, default=4,
                             help='output stride. Currently only supports 4.')
    self.parser.add_argument('--not_idaup', action='store_true')# F
    self.parser.add_argument('--num_classes', type=int, default= 1 )
    self.parser.add_argument('--backbone', default='dla34')
    self.parser.add_argument('--neck', default='dlaup')
    self.parser.add_argument('--msra_outchannel', type=int, default=256)
    self.parser.add_argument('--efficient_level', type=int, default=0)
    self.parser.add_argument('--prior_bias', type=float, default= -2.19) # -2.19

    # input res 
    self.parser.add_argument('--input_h', type=int, default=544, 
                             help='input height. -1 for default from dataset.')
    self.parser.add_argument('--input_w', type=int, default=1024, 
                             help='input width. -1 for default from dataset.')

    # train
    self.parser.add_argument('--num_workers', type=int, default=8,
                             help='dataloader threads. 0 for single-thread.')
    self.parser.add_argument('--batch_size', type=int, default= 4 ,
                             help='batch size')
    self.parser.add_argument('--optim', default='adam')
    self.parser.add_argument('--lr', type=float, default=1e-4, 
                             help='learning rate for batch size 32 - 1.25e-4.')
    self.parser.add_argument('--lr_step', type=str, default='20,40,50',
                             help='drop learning rate by 10.')
    self.parser.add_argument('--num_epochs', type=int, default=70,
                             help='total training epochs.')
    self.parser.add_argument('--val_intervals', type=int, default=5,
                             help='checkpoint saving interval; does not run validation.')
    self.parser.add_argument('--hm_disturb', type=float, default=0)
    self.parser.add_argument('--lost_disturb', type=float, default=0)
    self.parser.add_argument('--fp_disturb', type=float, default=0)
    self.parser.add_argument('--same_aug_pre', action='store_true')
    self.parser.add_argument('--max_frame_dist', type=int, default=3)
    
    # branches 
    self.parser.add_argument('--reid', type=boolean ,default=True )          
    self.parser.add_argument('--reid_weight', type=float, default=0.1)
    self.parser.add_argument('--tracking', type=boolean ,default=True)
    self.parser.add_argument('--tracking_weight', type=float, default=1)
    self.parser.add_argument('--ltrb_amodal',type=boolean ,default=True,)
    self.parser.add_argument('--ltrb_amodal_weight', type=float, default=0.1)
    self.parser.add_argument('--wh', action='store_true', help='use wh head for bbox size')
    self.parser.add_argument('--wh_weight', type=float, default=0.1,
                             help='loss weight for wh head.')
    self.parser.add_argument('--hm_weight', type=float, default=1,
                             help='loss weight for keypoint heatmaps.')
    self.parser.add_argument('--off_weight', type=float, default=1,
                             help='loss weight for keypoint local offsets.')
    self.parser.add_argument('--reset_hm', action='store_true')
    self.parser.add_argument('--reuse_hm', action='store_true')
    self.parser.add_argument('--add_05', action='store_true')
    self.parser.add_argument('--dense_reg', type=int, default=1, help='')

    # test
    self.parser.add_argument('--test_scales', type=str, default='1',
                             help='multi scale test augmentation.')
    self.parser.add_argument('--nms', action='store_true',
                             help='run nms in testing.')
    self.parser.add_argument('--K', type=int, default=200,
                             help='max number of output objects.') 
    self.parser.add_argument('--fix_short', type=int, default=-1)
    self.parser.add_argument('--keep_res', action='store_true',
                             help='keep the original resolution'
                                  ' during validation.')
    self.parser.add_argument('--out_thresh', type=float, default=-1,
                             help='')
    self.parser.add_argument('--depth_scale', type=float, default=1,
                             help='')
    self.parser.add_argument('--save_results', type=boolean, default=True)
    self.parser.add_argument('--ignore_loaded_cats', default='')
    self.parser.add_argument('--model_output_list', action='store_true',
                             help='Used when convert to onnx')
    self.parser.add_argument('--non_block_test', action='store_true')
    self.parser.add_argument('--vis_gt_bev', default='', help='')

    # dataset for aug
    self.parser.add_argument('--not_rand_crop', action='store_true',
                             help='not use the random crop data augmentation'
                                  'from CornerNet.')
    self.parser.add_argument('--not_max_crop', action='store_true',
                             help='used when the training dataset has'
                                  'inbalanced aspect ratios.')
    self.parser.add_argument('--shift', type=float, default=0,
                             help='when not using random crop, 0.1'
                                  'apply shift augmentation.')
    self.parser.add_argument('--scale', type=float, default=0,
                             help='when not using random crop, 0.4'
                                  'apply scale augmentation.')
    self.parser.add_argument('--aug_rot', type=float, default=0, 
                             help='probability of applying '
                                  'rotation augmentation.')
    self.parser.add_argument('--rotate', type=float, default=0,
                             help='when not using random crop'
                                  'apply rotation augmentation.')
    self.parser.add_argument('--flip', type=float, default=0.5,
                             help='probability of applying flip augmentation.')
    self.parser.add_argument('--no_color_aug', action='store_true',
                             help='not use the color augmenation '
                                  'from CornerNet')

    # Tracking
    self.parser.add_argument('--pre_hm', action='store_true')
    self.parser.add_argument('--zero_pre_hm', action='store_true')
    self.parser.add_argument('--pre_thresh', type=float, default=-1)
    self.parser.add_argument('--track_thresh', type=float, default=-1)
    self.parser.add_argument('--new_thresh', type=float, default=0.3)
    self.parser.add_argument('--hungarian', action='store_true', help='if false, using Greedy matching')
    self.parser.add_argument('--max_age', type=int, default=30 )
    self.parser.add_argument('--motion_cost_limit', type=float, default=0.75,
                             help='First-stage cost limit. Cost is 1-IoU, so 0.75 keeps IoU >= 0.25.')
    self.parser.add_argument('--app_cost_limit', type=float, default=0.35,
                             help='Second-stage cosine-distance limit. 0.35 keeps similarity >= 0.65.')
    self.parser.add_argument('--app_iou_gate', type=float, default=0.9,
                             help='Drop appearance pairs whose IoU cost exceeds this value.')
    self.parser.add_argument('--confirm_cost_limit', type=float, default=0.7,
                             help='Third-stage cost limit for unconfirmed tracks.')
    self.parser.add_argument('--box_ema_long', type=float, default=0.65,
                             help='Box EMA weight after the tracklet exceeds box_ema_switch.')
    self.parser.add_argument('--box_ema_short', type=float, default=0.35,
                             help='Box EMA weight for shorter tracklets.')
    self.parser.add_argument('--box_ema_switch', type=int, default=5)
    self.parser.add_argument('--buffer_score', type=float, default=0.55,
                             help='Unconfirmed tracks at or above this score are kept for one frame.')
    self.parser.add_argument('--reid_momentum', type=float, default=0.4,
                             help='gamma in the confidence-adaptive appearance update.')
    self.parser.add_argument('--inital_thresh', type=float, default=0.5)
    self.parser.add_argument('--test_device', type=int, default=0, help='a test device is selected .')
    
    # loss
    self.parser.add_argument('--reg_loss', default='l1',
                             help='regression loss: sl1 | l1 | l2')
    self.parser.add_argument('--amodel_offset_weight', type=float, default=1,
                             help='Please forgive the typo.')
    self.parser.add_argument('--focalLoss', type=str, default='FocalLoss',choices=['FastFocalLoss','FocalLoss'],
                             help='focalLoss: FastFocalLoss | FocalLoss ')
    self.parser.add_argument('--iouloss', type=boolean, default=True, help='use iouloss or not')
    self.parser.add_argument('--use_siou', type=boolean, default=True, help='use siou or not')
    self.parser.add_argument('--theta', type=int, default=4, choices=[2,3,4,5,6], help='hyper-parameter used in siou_loss')

    # custom dataset
    self.parser.add_argument('--custom_dataset_img_path', default='')
    self.parser.add_argument('--custom_dataset_ann_path', default='')
    
    # test data sets
    self.parser.add_argument('--test_data_root', default='')
    
    

  def parse(self, args=''):
    if args == '':#T
      opt = self.parser.parse_args()
    else:
      opt = self.parser.parse_args(args)

    if opt.resume and not opt.load_model:
      self.parser.error('--resume requires --load_model')
    if opt.resume and opt.allow_partial_load:
      self.parser.error('--resume requires a complete checkpoint, not --allow_partial_load')
    if opt.head_kernel < 1 or opt.head_kernel % 2 == 0:
      self.parser.error('--head_kernel must be a positive odd integer')
    if opt.num_head_conv < 0 or opt.num_head_conv > 4:
      self.parser.error('--num_head_conv must be between 0 and 4')
    if opt.head_conv != -1 and opt.head_conv < 1:
      self.parser.error('--head_conv must be positive or -1 for the default width')
  
    opt.test_scales = [float(i) for i in opt.test_scales.split(',')]
    opt.save_imgs = [i for i in opt.save_imgs.split(',')] \
      if opt.save_imgs != '' else []
    opt.ignore_loaded_cats = \
      [int(i) for i in opt.ignore_loaded_cats.split(',')] \
      if opt.ignore_loaded_cats != '' else []# --- opt.save_imgs = []  opt.ignore_loaded_cats = []


    opt.pre_img = False
    if 'tracking' in opt.task:
      if opt.local_rank in [-1,0]:
        lg.info('Running tracking')
      opt.tracking = True
      opt.out_thresh = max(opt.track_thresh, opt.out_thresh)
      opt.pre_thresh = max(opt.track_thresh, opt.pre_thresh)
      opt.new_thresh = max(opt.track_thresh, opt.new_thresh)
      opt.pre_img = True# T
      if opt.local_rank in [-1,0]:
        lg.info('Using tracking threshold for out threshold: {}.'.format(opt.track_thresh))


    opt.fix_res = not opt.keep_res
    if opt.local_rank in [-1,0]:
      lg.info('Fix size testing.' if opt.fix_res else 'Keep resolution testing.')

    if opt.head_conv == -1: # init default head_conv
      opt.head_conv = 256 if 'dla' in opt.arch else 64 # dla-34，opt.head_conv = 256 

    opt.pad = 127 if 'hourglass' in opt.arch else 31 
    opt.num_stacks = 2 if opt.arch == 'hourglass' else 1 

    # log dirs
    opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')# get file path
    opt.data_dir = os.path.join(opt.root_dir, 'data')
    opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
    opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
    opt.debug_dir = os.path.join(opt.save_dir, 'debug')
      
    return opt


  def update_res_info_and_set_heads(self, opt):
    opt.output_h = opt.input_h // opt.down_ratio # opt.output_h(w)：output resolution
    opt.output_w = opt.input_w // opt.down_ratio
    opt.input_res = max(opt.input_h, opt.input_w) #  input_resolution
    opt.output_res = max(opt.output_h, opt.output_w) #  output_resolution
    
    # build heads
    opt.heads = {'hm': opt.num_classes, 'reg': 2}
    if 'tracking' in opt.task:# T
      opt.heads.update({'tracking': 2}) 
      # opt.heads = {'hm', 'reg', 'tracking', 'ltrb_amodal'} 
    if opt.ltrb_amodal:
      opt.heads.update({'ltrb_amodal': 4})
    if opt.wh:
      opt.heads.update({'wh': 2})
    if opt.reid:
      opt.heads.update({'reid': 128})
      
    # the loss weight per head
    weight_dict = {'hm': opt.hm_weight, 'reg': opt.off_weight, 
                   'amodel_offset': opt.amodel_offset_weight,
                   'reid': opt.reid_weight, 'tracking': opt.tracking_weight,
                   'ltrb_amodal': opt.ltrb_amodal_weight,
                   'wh': opt.wh_weight } # use for loss           
    opt.weights = {head: weight_dict[head] for head in opt.heads}# use for loss
    
    for head in opt.weights:
      if opt.weights[head] == 0:
        del opt.heads[head] # delete branch which have the loss weight == 0 
    opt.head_conv = {head: [opt.head_conv \
      for i in range(opt.num_head_conv if head != 'reg' else 1)] for head in opt.heads}
    # ouput channel 256
    if opt.local_rank in [-1,0]:
      lg.info('input h: {},  w: {}'.format(opt.input_h, opt.input_w))
      lg.info('heads: {}'.format(opt.heads))
      lg.info('weights: {}'.format(opt.weights))
      lg.info('head conv: {}'.format(opt.head_conv))

    return opt

  def init(self,):
    # only used in test
    opt = self.parse()
    opt = self.update_res_info_and_set_heads(opt)
    return opt
