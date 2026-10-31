"""Regression tests for data identity, evaluation frames and Q-Attention."""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/lib'),str(ROOT/'src/tools')]
from mot_format import convert
from prepare_trackeval import prepare
from opts import opts
from model.model import create_model, load_model
from model.networks.base_model import QAttention
from model.utils import fuse_model
from tracking_utils.evaluation import Evaluator
from dataset.datasets.custom_dataset import CustomDataset


def fixture(root, frames=4, names=('SEQ_A', 'SEQ_B')):
    for split in ('train','val','test'):
        for name in names:
            seq=root/split/name
            (seq/'img1').mkdir(parents=True)
            (seq/'gt').mkdir()
            rows=[]
            for frame in range(1,frames+1):
                cv2.imwrite(str(seq/'img1'/f'{frame:08d}.jpg'),np.full((64,64,3),120,dtype=np.uint8))
                for tid in (1,2):
                    rows.append(f'{frame},{tid},{tid*15},15,10,10,1,1,1\n')
            (seq/'gt/gt.txt').write_text(''.join(rows))


def options(extra=()):
    parser=opts()
    return parser.update_res_info_and_set_heads(parser.parse([
        '--use_qattn','--wh','--pre_hm','--no_pretrained',
        '--input_h','128','--input_w','128']+list(extra)))


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='gatemot_test_')
        self.root=Path(self.tmp.name)
        self.names=('SEQ_A-FRCNN','SEQ_B-FRCNN')
        fixture(self.root,names=self.names)

    def tearDown(self):
        self.tmp.cleanup()

    def test_identity_is_stable_and_video_specific(self):
        for dataset in ('bee','mot17','mot20','sportsmot'):
            convert(dataset,self.root,['train'])
            data=json.loads((self.root/'annotations/train.json').read_text())
            video={x['id']:x['video_id'] for x in data['images']}
            pairs={}
            for ann in data['annotations']:
                key=(video[ann['image_id']],ann['source_track_id'])
                pairs.setdefault(key,set()).add(ann['track_id'])
            self.assertTrue(all(len(v)==1 for v in pairs.values()))
            self.assertEqual(len({next(iter(v)) for v in pairs.values()}),4)
        opt=options(['--custom_dataset_img_path',str(self.root/'train'),
                     '--custom_dataset_ann_path',str(self.root/'annotations/train.json')])
        # No access to GT is needed after conversion, including the unused split.
        for path in self.root.glob('*/SEQ_*/gt/gt.txt'):
            path.unlink()
        dataset=CustomDataset(opt,'train')
        self.assertEqual(opt.nid,4)
        self.assertEqual(len(dataset),8)

    def test_half_split_and_trackeval_bundle(self):
        convert('bee',self.root,['halftrain','halfval'])
        train=json.loads((self.root/'annotations/halftrain.json').read_text())
        val=json.loads((self.root/'annotations/halfval.json').read_text())
        self.assertFalse({x['file_name'] for x in train['images']} & {x['file_name'] for x in val['images']})
        tracks=self.root/'tracks'
        tracks.mkdir()
        for name in self.names:
            (tracks/(name+'.txt')).write_text((self.root/'train'/name/'gt/gt.txt').read_text())
        prepare(self.root/'train',self.root/'annotations/halfval.json',tracks,self.root/'bundle','bee24')
        lines=(self.root/'bundle/gt'/self.names[0]/'gt/gt.txt').read_text().splitlines()
        self.assertEqual(len(lines),4)
        self.assertEqual({line.split(',')[0] for line in lines},{'1','2'})

    def test_default_main_splits_do_not_read_test_ground_truth(self):
        # Hidden test labels must never enter the converted training manifests.
        # Deliberately invalid files make an accidental test-GT read fail.
        for path in (self.root/'test').glob('*/gt/gt.txt'):
            path.write_text('synthetic test labels are reserved for evaluation\n')
        for dataset in ('bee','mot17','mot20','sportsmot'):
            with self.subTest(dataset=dataset):
                shutil.rmtree(self.root/'annotations',ignore_errors=True)
                convert(dataset,self.root)
                expected={'train','val','test'} if dataset=='sportsmot' else {'train','test'}
                self.assertEqual({p.stem for p in (self.root/'annotations').glob('*.json')},expected)
                train=json.loads((self.root/'annotations/train.json').read_text())
                test=json.loads((self.root/'annotations/test.json').read_text())
                self.assertEqual(len(train['images']),8)
                self.assertEqual(len(train['annotations']),16)
                self.assertEqual({im['frame_id'] for im in train['images']},{1,2,3,4})
                self.assertEqual(len(test['images']),8)
                self.assertEqual(test['annotations'],[])
                self.assertEqual(test['conversion']['source_split'],'test')

    def test_three_dataset_half_pipelines_keep_frames_and_history_separate(self):
        # Only the training directory is available. SEQ_A is even, SEQ_B is odd.
        shutil.rmtree(self.root/'val')
        shutil.rmtree(self.root/'test')
        seq=self.root/'train'/self.names[1]
        cv2.imwrite(str(seq/'img1/00000005.jpg'),np.full((64,64,3),125,dtype=np.uint8))
        with (seq/'gt/gt.txt').open('a') as handle:
            for tid in (1,2):
                handle.write(f'5,{tid},{tid*15},15,10,10,1,1,1\n')
        tracks=self.root/'tracks'
        tracks.mkdir()
        for name in self.names:
            (tracks/(name+'.txt')).write_text((self.root/'train'/name/'gt/gt.txt').read_text())
        env=dict(os.environ,DATA_ROOT=str(self.root),
                 PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''))
        for dataset in ('bee24','mot17','mot20'):
            with self.subTest(dataset=dataset):
                subprocess.run(['bash','scripts/prepare_data.sh',dataset,
                                '--splits','halftrain','halfval','train'],
                               cwd=str(ROOT),env=env,check=True,capture_output=True,text=True)
                data={split:json.loads((self.root/'annotations'/f'{split}.json').read_text())
                      for split in ('halftrain','halfval','train')}
                paths={split:{im['file_name'] for im in item['images']} for split,item in data.items()}
                self.assertFalse(paths['halftrain'] & paths['halfval'])
                self.assertEqual(paths['halftrain'] | paths['halfval'],paths['train'])
                for split,loader_split in (('halftrain','train'),('halfval','val')):
                    item=data[split]
                    self.assertEqual(item['conversion']['half_split_policy'],'ceil')
                    images={im['id']:im for im in item['images']}
                    for video in item['videos']:
                        frames=sorted((im for im in images.values() if im['video_id']==video['id']),
                                      key=lambda im:im['frame_id'])
                        length=4 if video['file_name']==self.names[0] else 5
                        boundary=(length+1)//2
                        expected=list(range(1,boundary+1) if split=='halftrain'
                                      else range(boundary+1,length+1))
                        self.assertEqual([im['frame_id'] for im in frames],expected)
                        self.assertEqual(frames[0]['prev_image_id'],-1)
                        self.assertEqual(frames[-1]['next_image_id'],-1)
                        for index,im in enumerate(frames):
                            self.assertEqual(int(Path(im['file_name']).stem),im['frame_id'])
                            if index:
                                self.assertEqual(im['prev_image_id'],frames[index-1]['id'])
                            if index+1<len(frames):
                                self.assertEqual(im['next_image_id'],frames[index+1]['id'])
                    self.assertTrue(all(ann['image_id'] in images for ann in item['annotations']))
                    opt=options(['--custom_dataset_img_path',str(self.root/'train'),
                                 '--custom_dataset_ann_path',str(self.root/'annotations'/f'{split}.json')])
                    loader=CustomDataset(opt,loader_split)
                    # Check the actual previous-frame loader, not only JSON links.
                    with patch.object(loader,'_load_image_anns',wraps=loader._load_image_anns) as read:
                        for im in item['images']:
                            loader._load_pre_data(im['video_id'],im['frame_id'])
                            loaded=images[read.call_args.args[0]]
                            self.assertEqual(loaded['video_id'],im['video_id'])
                            self.assertIn(loaded['file_name'],paths[split])
                output=self.root/('bundle_'+dataset)
                prepare(self.root/'train',self.root/'annotations/halfval.json',tracks,output,dataset)
                for name,count in ((self.names[0],2),(self.names[1],2)):
                    gt=(output/'gt'/name/'gt/gt.txt').read_text().splitlines()
                    pred=(output/'trackers'/(name+'.txt')).read_text().splitlines()
                    self.assertEqual(len(gt),2*count)
                    self.assertEqual(gt,pred)
                    self.assertEqual({int(row.split(',')[0]) for row in gt},set(range(1,count+1)))

    def test_empty_prediction_frames_count_as_misses(self):
        prediction=self.root/'pred.txt'
        prediction.write_text('1,1,15,15,10,10,1,-1,-1,-1\n')
        evaluator=Evaluator(str(self.root/'train'),self.names[0],'mot')
        import motmetrics as mm
        summary=mm.metrics.create().compute(evaluator.eval_file(str(prediction)),metrics=['num_frames','num_misses'])
        self.assertEqual(int(summary.iloc[0]['num_frames']),4)
        self.assertEqual(int(summary.iloc[0]['num_misses']),7)
        summary=mm.metrics.create().compute(evaluator.eval_file(str(prediction),frame_ids=[3,4]),metrics=['num_frames','num_misses'])
        self.assertEqual(int(summary.iloc[0]['num_frames']),2)
        self.assertEqual(int(summary.iloc[0]['num_misses']),4)

    def test_qattention_backward_and_bn_fusion(self):
        layer=QAttention(64,256).eval()
        x=torch.randn(2,64,8,8,requires_grad=True)
        expected=layer(x)
        torch.testing.assert_close(expected,fuse_model(copy.deepcopy(layer))(x),rtol=1e-4,atol=1e-5)
        expected.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_task_heads_have_independent_qattention(self):
        opt=options()
        model=create_model(opt.arch,opt.heads,opt.head_conv,opt)
        self.assertEqual(set(model.heads),{'hm','reg','tracking','ltrb_amodal','wh','reid'})
        gates=[]
        for name in model.heads:
            head=getattr(model,name)
            self.assertIsInstance(head[0],QAttention)
            self.assertGreater(opt.weights[name],0)
            gates.append(head[0].conv3.weight)
        self.assertEqual(len({id(gate) for gate in gates}),len(model.heads))

    def test_affine_augmentation_accepts_scalar_and_vector_scale(self):
        from dataset.generic_dataset import GenericDataset
        dataset=GenericDataset()
        dataset.opt=options(['--not_rand_crop'])
        for scale in (64.0,np.array([64.,48.],dtype=np.float32)):
            center,aug,rotation=dataset._get_aug_param(np.array([32.,32.]),scale,64,64)
            self.assertTrue(np.isfinite(center).all())
            self.assertTrue(np.isfinite(aug))

    def test_checkpoint_mismatch_is_rejected(self):
        opt=options()
        model=torch.nn.Linear(2,2)
        path=self.root/'wrong.pth'
        torch.save({'epoch':1,'state_dict':{'weight':torch.zeros(3,3)}},path)
        with self.assertRaises(RuntimeError):
            load_model(model,str(path),opt)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
