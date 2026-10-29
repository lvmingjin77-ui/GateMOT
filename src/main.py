"""Single-device or torchrun training with explicit run provenance."""
import _init_paths
import hashlib
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from lib.opts import opts
from lib.logger import Logger
from lib.trainer import Trainer
from lib.dataset.dataset_factory import get_dataset
from lib.model.model import create_model, save_model


def main(opt):
    opt.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    opt.global_rank = int(os.environ.get('RANK', '0'))
    opt.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    use_cuda = opt.device != 'cpu' and torch.cuda.is_available()
    if opt.device == 'cuda' and not use_cuda:
        raise RuntimeError('CUDA was requested but is unavailable')
    opt.device = torch.device('cuda',opt.local_rank) if use_cuda else torch.device('cpu')
    if use_cuda:
        torch.cuda.set_device(opt.local_rank)
    if opt.world_size > 1:
        dist.init_process_group('nccl' if use_cuda else 'gloo')
    if opt.batch_size < opt.world_size or opt.batch_size % opt.world_size:
        raise ValueError('Global batch size must be a positive multiple of WORLD_SIZE')
    opt.batch_size_per_gpu = opt.batch_size // opt.world_size
    random.seed(opt.seed + opt.global_rank)
    np.random.seed(opt.seed + opt.global_rank)
    torch.manual_seed(opt.seed + opt.global_rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    directory = Path(opt.save_dir)
    if directory.exists() and list(directory.glob('model_*.pth')) and not opt.resume:
        raise FileExistsError('Experiment already contains checkpoints. Choose a new --exp_id or use --resume.')
    dataset = get_dataset(opt.dataset)(opt, 'train')
    model = create_model(opt.arch,opt.heads,opt.head_conv,opt)
    sampler = torch.utils.data.DistributedSampler(dataset,seed=opt.seed) if opt.world_size>1 else None
    loader = torch.utils.data.DataLoader(dataset,batch_size=opt.batch_size_per_gpu,
        sampler=sampler,shuffle=sampler is None,num_workers=opt.num_workers,pin_memory=use_cuda)
    trainer = Trainer(opt,model)
    start = trainer.resume_epoch if opt.load_model and opt.resume else 0
    steps = [int(x) for x in opt.lr_step.split(',') if x]
    if opt.global_rank == 0:
        logger = Logger(opt)
        metadata = {k:str(v) if isinstance(v,torch.device) else v for k,v in vars(opt).items()}
        metadata.update(command=sys.argv,torch_version=torch.__version__,
            cuda_version=torch.version.cuda,annotation_sha256=hashlib.sha256(Path(opt.custom_dataset_ann_path).read_bytes()).hexdigest(),
            model_parameters=sum(p.numel() for p in model.parameters()),
            training_parameters=sum(p.numel() for p in trainer.model_with_loss.parameters()))
        source_root=Path(__file__).resolve().parent
        metadata['source_sha256']={str(p.relative_to(source_root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_root.rglob('*.py')}
        (directory/('run_config_resume_{}.json'.format(start) if opt.resume else 'run_config.json')).write_text(json.dumps(metadata,indent=2))
    for epoch in range(start+1,opt.num_epochs+1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        lr=opt.lr * (0.1 ** sum(epoch>s for s in steps))
        if opt.warmup_epochs and epoch<=opt.warmup_epochs:
            lr *= epoch / opt.warmup_epochs
        for group in trainer.optimizer.param_groups:
            group['lr']=lr
        stats,_=trainer.train(epoch,loader)
        if opt.global_rank == 0:
            logger.write('epoch {} {}\n'.format(epoch,stats))
            for key,value in stats.items():
                logger.scalar_summary('train_'+key,value,epoch)
            save_model(str(directory/'model_last.pth'),epoch,trainer.model_with_loss,trainer.optimizer)
            if epoch==opt.num_epochs or (opt.val_intervals>0 and epoch%opt.val_intervals==0):
                save_model(str(directory/('model_{}.pth'.format(epoch))),epoch,trainer.model_with_loss,trainer.optimizer)
    if opt.global_rank == 0:
        logger.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    parser=opts()
    main(parser.update_res_info_and_set_heads(parser.parse()))
