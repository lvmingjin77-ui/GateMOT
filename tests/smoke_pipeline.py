"""Tiny train/resume/inference exercise. Never measures tracking accuracy."""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from test_release import fixture, ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    args=parser.parse_args()
    root=Path(tempfile.mkdtemp(prefix='gatemot_smoke_'))
    fixture(root)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
    def run(argv):
        subprocess.run([sys.executable]+argv,cwd=str(ROOT),env=env,check=True,timeout=240)
    run(['src/tools/bee_to_coco.py','--data-root',str(root)])
    common=['--head_type','qattn','--wh','--pre_hm','--device',args.device,
        '--input_h','128','--input_w','128','--K','10','--custom_dataset_img_path',str(root/'train'),
        '--custom_dataset_ann_path',str(root/'annotations/halftrain.json')]
    training=['src/main.py']+common+['--no_pretrained','--exp_id',str(root/'run'),
        '--num_epochs','1','--batch_size','2','--num_workers','0','--not_rand_crop','--flip','0','--no_color_aug']
    run(training)
    checkpoint=root/'run/model_last.pth'
    assert checkpoint.is_file()
    run(training+['--resume','--load_model',str(checkpoint),'--num_epochs','2'])
    run(['src/test.py']+common+['--custom_dataset_ann_path',str(root/'annotations/halfval.json'),
        '--load_model',str(checkpoint),'--track_thresh','0.4','--hungarian','--output_dir',str(root/'tracks')])
    metadata=json.loads((root/'tracks/evaluation_config.json').read_text())
    assert all(frames==[3,4] for frames in metadata['selected_frames'].values())
    assert (root/'tracks/clear_summary.csv').is_file()
    print('SMOKE_PIPELINE_OK',root)


if __name__=='__main__':
    main()
