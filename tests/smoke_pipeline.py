"""Exercise the public scripts on synthetic temporal or official data splits."""
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
    parser.add_argument('--trackeval', help='Optional TrackEval source directory for metric evaluation')
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix='gatemot_smoke_'))
    fixture(root)
    official_split = args.dataset == 'sportsmot'
    train_split = 'train' if official_split else 'halftrain'
    eval_split = 'val' if official_split else 'halfval'
    expected_frames = [1,2,3,4] if official_split else [3,4]
    expected_arch = 'dla60' if official_split else 'dla34'
    # Neither workflow requires a test split. Temporal halves also need no val/.
    shutil.rmtree(root/'test')
    if not official_split:
        shutil.rmtree(root/'val')
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg', DATA_ROOT=str(root),
        PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''),
        ANN_NAME=train_split, EXP_ID=str(root/'run'), NUM_GPUS='1',
        GLOBAL_BATCH='2', NUM_EPOCHS='1', HEAD_TYPE='qattn', SEED='317',
        SPLIT=eval_split, DATASET=args.dataset, FUSE='0')

    def run(argv):
        subprocess.run(argv, cwd=str(ROOT), env=env, check=True, timeout=240)

    run(['bash', 'scripts/prepare_data.sh', args.dataset, '--splits', train_split, eval_split])
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
    env['LOAD_MODEL'] = str(checkpoint)
    run(['bash', 'scripts/test.sh', args.dataset] + common + ['--output_dir', str(root/'tracks')])
    metadata = json.loads((root/'tracks/evaluation_config.json').read_text())
    assert metadata['arch'] == expected_arch
    assert Path(metadata['custom_dataset_ann_path']) == root/'annotations'/(eval_split+'.json')
    assert Path(metadata['custom_dataset_img_path']) == root/('val' if official_split else 'train')
    assert set(metadata['selected_frames']) == {'SEQ_A','SEQ_B'}
    assert all(frames == expected_frames for frames in metadata['selected_frames'].values())
    assert (root/'tracks/clear_summary.csv').is_file()
    if args.trackeval:
        env.update(TRACK_EVAL=str(Path(args.trackeval).resolve()),
                   TRACKERS_FOLDER=str(root/'tracks'), EVAL_DIR=str(root/'evaluation'))
        run(['bash', 'scripts/eval_hota.sh'])
        summary = (root/'evaluation/trackers/pedestrian_summary.txt').read_text().splitlines()
        metrics = dict(zip(summary[0].split(),summary[1].split()))
        assert int(float(metrics['GT_Dets'])) == 2 * 2 * len(expected_frames)
    print('SMOKE_PIPELINE_OK', args.dataset, root)


if __name__ == '__main__':
    main()
