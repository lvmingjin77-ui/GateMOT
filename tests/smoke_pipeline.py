"""Exercise public train-to-test scripts on tiny synthetic data, not paper metrics.

The optional validation protocol uses temporal halves for BEE24/MOT17/MOT20
and the official train/val directories for SportsMOT. These smoke runs use
random initialization and 128x128 inputs only to verify the executable workflow.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from test_release import fixture, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--dataset', default='bee24', choices=['bee24', 'mot17', 'mot20', 'sportsmot'])
    parser.add_argument('--protocol', default='main', choices=['main', 'validation'],
                        help='main: full train to test; validation: optional development split')
    parser.add_argument('--trackeval', help='Optional TrackEval source directory for metric evaluation')
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix='gatemot_smoke_'))
    fixture(root)
    official_split = args.dataset == 'sportsmot'
    validation = args.protocol == 'validation'
    train_split = 'halftrain' if validation and not official_split else 'train'
    eval_split = ('val' if official_split else 'halfval') if validation else 'test'
    expected_frames = [3,4] if eval_split == 'halfval' else [1,2,3,4]
    training_frames = [1,2] if train_split == 'halftrain' else [1,2,3,4]
    expected_arch = 'dla60' if official_split else 'dla34'
    if not official_split:
        shutil.rmtree(root/'val')
    if validation:
        # Development validation does not depend on access to a test split.
        shutil.rmtree(root/'test')
    # Synthetic test labels are withheld through conversion and both training
    # epochs, then restored only for evaluation. Official hidden labels are not
    # needed for conversion, training, or prediction on a real test split.
    test_gt={path:path.read_text() for path in (root/'test').glob('*/gt/gt.txt')}
    for path in test_gt:
        path.unlink()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
        MPLBACKEND='Agg', DATA_ROOT=str(root),
        PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''),
        EXP_ID=str(root/'run'), NUM_GPUS='1', GLOBAL_BATCH='2', NUM_EPOCHS='1',
        SEED='317', DATASET=args.dataset, FUSE='0')
    # Exercise the public main defaults, regardless of the caller's shell.
    for name in ('ANN_NAME','SPLIT'):
        env.pop(name,None)
    if validation:
        env.update(ANN_NAME=train_split,SPLIT=eval_split)

    def run(argv):
        subprocess.run(argv, cwd=str(ROOT), env=env, check=True, timeout=240)

    prepare_command = ['bash', 'scripts/prepare_data.sh', args.dataset]
    if validation:
        prepare_command += ['--splits',train_split,eval_split]
    run(prepare_command)
    train_data=json.loads((root/'annotations'/(train_split+'.json')).read_text())
    eval_data=json.loads((root/'annotations'/(eval_split+'.json')).read_text())
    assert {im['frame_id'] for im in train_data['images']} == set(training_frames)
    assert len(train_data['images']) == 2 * len(training_frames)
    assert len(train_data['annotations']) == 2 * 2 * len(training_frames)
    if not validation:
        assert eval_data['annotations'] == []
        expected_splits={'train','val','test'} if official_split else {'train','test'}
        assert {path.stem for path in (root/'annotations').glob('*.json')} == expected_splits
    common = ['--device', args.device, '--input_h', '128', '--input_w', '128', '--K', '10']
    training = ['bash', 'scripts/train.sh', args.dataset] + common + [
        '--no_pretrained', '--num_workers', '0', '--not_rand_crop', '--flip', '0',
        '--aug_rot', '0', '--no_color_aug']
    run(training)
    checkpoint = root/'run/model_last.pth'
    assert checkpoint.is_file()
    train_config = json.loads((root/'run/run_config.json').read_text())
    assert train_config['arch'] == expected_arch
    assert Path(train_config['custom_dataset_ann_path']) == root/'annotations'/(train_split+'.json')
    assert Path(train_config['custom_dataset_img_path']) == root/'train'
    run(training + ['--resume', '--load_model', str(checkpoint), '--num_epochs', '2'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert saved['epoch'] == 2
    del saved
    resume_config = json.loads((root/'run/run_config_resume_1.json').read_text())
    assert resume_config['arch'] == expected_arch and resume_config['num_epochs'] == 2
    assert Path(resume_config['custom_dataset_ann_path']) == root/'annotations'/(train_split+'.json')
    for path,contents in test_gt.items():
        path.write_text(contents)
    env['LOAD_MODEL'] = str(checkpoint)
    run(['bash', 'scripts/test.sh', args.dataset] + common + ['--output_dir', str(root/'tracks')])
    metadata = json.loads((root/'tracks/evaluation_config.json').read_text())
    assert metadata['arch'] == expected_arch
    assert Path(metadata['custom_dataset_ann_path']) == root/'annotations'/(eval_split+'.json')
    eval_images = 'train' if eval_split == 'halfval' else eval_split
    assert Path(metadata['custom_dataset_img_path']) == root/eval_images
    assert set(metadata['selected_frames']) == {'SEQ_A','SEQ_B'}
    assert all(frames == expected_frames for frames in metadata['selected_frames'].values())
    assert (root/'tracks/clear_summary.csv').is_file()
    metrics={}
    if args.trackeval:
        env.update(TRACK_EVAL=str(Path(args.trackeval).resolve()),
                   TRACKERS_FOLDER=str(root/'tracks'), EVAL_DIR=str(root/'evaluation'))
        run(['bash', 'scripts/eval_hota.sh'])
        summary = (root/'evaluation/trackers/pedestrian_summary.txt').read_text().splitlines()
        metrics = dict(zip(summary[0].split(),summary[1].split()))
        assert int(float(metrics['GT_Dets'])) == 2 * 2 * len(expected_frames)
    proof=dict(dataset=args.dataset,protocol=args.protocol,arch=expected_arch,
               training_split=train_split,evaluation_split=eval_split,
               training_frames=training_frames,selected_frames=metadata['selected_frames'],
               checkpoint_epoch=2,test_gt_withheld_until_training_finished=not validation,
               gt_detections=int(float(metrics['GT_Dets'])) if metrics else None,
               synthetic=True,fixture=str(root))
    (root/'smoke_summary.json').write_text(json.dumps(proof,indent=2))
    print('SMOKE_PIPELINE_OK', args.dataset, args.protocol, root)


if __name__ == '__main__':
    main()
