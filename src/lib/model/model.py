from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
import os
from loguru import logger as lg
from .networks.dla import DLASeg

_network_factory = {
  'dla': DLASeg,
}

def create_model(arch, head, head_conv, opt=None):
  num_layers = int(arch[3:5]) # dla34 dla46x_c
  arch = arch[:3] # 'dla'
  model_class = _network_factory[arch]# model_class == DLASeg
  model = model_class(num_layers, heads=head, head_convs=head_conv, opt=opt)
  return model

def load_model(model, model_path, opt, optimizer=None):
  if opt.allow_partial_load and optimizer is None:
    raise ValueError('Inference requires a complete checkpoint. --allow_partial_load is only for training initialization.')
  if opt.resume and optimizer is not None and opt.allow_partial_load:
    raise ValueError('Resume requires a complete, matching checkpoint')
  # Load only checkpoints from trusted sources. Optimizer state uses pickle.
  checkpoint = torch.load(model_path, map_location='cpu')
  source = checkpoint['state_dict']
  target = model.state_dict()
  training = optimizer is not None
  state = {}
  for key, value in source.items():
    if key.startswith('module.'):
      key = key[7:]
    if not training:
      if key.startswith('loss.'):
        continue
      if key.startswith('model.'):
        key = key[6:]
    elif not key.startswith(('model.', 'loss.')):
      key = 'model.' + key
    state[key] = value
  mismatched = [key for key in state if key in target and state[key].shape != target[key].shape]
  missing = sorted(set(target)-set(state))
  unexpected = sorted(set(state)-set(target))
  if mismatched or missing or unexpected:
    message = 'Checkpoint mismatch. Missing={}, unexpected={}, shape mismatches={}'.format(missing,unexpected,mismatched)
    if not opt.allow_partial_load:
      raise RuntimeError(message + '. Match the backbone, heads, and --use_qattn setting (enabled for Q-Attention, omitted for Conv). Partial loading is for explicit training initialization only.')
    lg.warning(message)
    state = {k:v for k,v in state.items() if k in target and k not in mismatched}
  model.load_state_dict(state, strict=not opt.allow_partial_load)
  epoch = 0
  if training and opt.resume:
    if opt.allow_partial_load:
      raise ValueError('Resume requires a complete, matching checkpoint')
    if 'optimizer' not in checkpoint:
      raise ValueError('Checkpoint has no optimizer state to resume')
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
  lg.info('Loaded {} at epoch {}', model_path, checkpoint.get('epoch','unknown'))
  return (model,optimizer,epoch) if training else model

def save_model(path, epoch, model, optimizer=None):
  '''
  path, 
  epoch, 
  trainer.model_with_loss, 
  trainer.optimizer

  '''
  if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
    model_state_dict = model.module.state_dict()
  else:
    model_state_dict = model.state_dict()
    
  data = {'epoch': epoch, 'state_dict': model_state_dict}
  
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)

class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
  def _check_input_dim(self, input):
      # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
      # is this method that is overwritten by the sub-class
      # This original goal of this method was for tensor sanity checks
      # If you're ok bypassing those sanity checks (eg. if you trust your inference
      # to provide the right dimensional inputs), then you can just use this method
      # for easy conversion from SyncBatchNorm
      # (unfortunately, SyncBatchNorm does not store the original class - if it did
      #  we could return the one that was originally created)
      return

def revert_sync_batchnorm(module):
  # this is very similar to the function that it is trying to revert:
  # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
  module_output = module
  if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
    new_cls = BatchNormXd
    module_output = BatchNormXd(module.num_features,
                                            module.eps, module.momentum,
                                            module.affine,
                                            module.track_running_stats)
    if module.affine:
      with torch.no_grad():
        module_output.weight = module.weight
        module_output.bias = module.bias
    module_output.running_mean = module.running_mean
    module_output.running_var = module.running_var
    module_output.num_batches_tracked = module.num_batches_tracked
    if hasattr(module, "qconfig"):
      module_output.qconfig = module.qconfig
  for name, child in module.named_children():
    module_output.add_module(name, revert_sync_batchnorm(child))
  del module
  return module_output
