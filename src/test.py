"""Track exactly the sequences and frames listed in the annotation manifest."""
import _init_paths
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import cv2
import torch
import motmetrics as mm
from lib.opts import opts
from lib.detector import Detector
from lib.tracking_utils.evaluation import Evaluator


def main(opt):
    annotation = Path(opt.custom_dataset_ann_path)
    data = json.loads(annotation.read_text())
    sequences = {v['id']: v['file_name'] for v in data['videos']}
    images = defaultdict(list)
    for image in data['images']:
        images[image['video_id']].append(image)
    if not images:
        raise ValueError('The selected annotation manifest contains no images')
    checksum=hashlib.sha256(Path(opt.load_model).read_bytes()).hexdigest()
    output = Path(opt.output_dir or ('results/{}_{}_{}'.format(annotation.stem,Path(opt.load_model).stem,checksum[:8])))
    output.mkdir(parents=True,exist_ok=True)
    if list(output.glob('*.txt')):
        raise FileExistsError('Result directory contains tracking outputs. Choose a fresh --output_dir.')
    detector = Detector(opt)
    accs,names=[],[]
    for video_id,frames in images.items():
        sequence=sequences[video_id]
        frames.sort(key=lambda im: im['frame_id'])
        detector.reset_tracking()
        with (output/(sequence+'.txt')).open('w') as handle:
            for info in frames:
                path=Path(opt.custom_dataset_img_path)/info['file_name']
                image=cv2.imread(str(path))
                if image is None:
                    raise ValueError('Unreadable image {}'.format(path))
                result=detector.run(image)
                for track in result['results']:
                    x1,y1,x2,y2=track.tlbr
                    handle.write('{},{},{:.4f},{:.4f},{:.4f},{:.4f},1,-1,-1,-1\n'.format(info['frame_id'],track.track_id,x1,y1,x2-x1,y2-y1))
        gt=Path(opt.custom_dataset_img_path)/sequence/'gt/gt.txt'
        if gt.is_file():
            evaluator=Evaluator(opt.custom_dataset_img_path,sequence,'mot')
            accs.append(evaluator.eval_file(str(output/(sequence+'.txt')),frame_ids=[im['frame_id'] for im in frames]))
            names.append(sequence)
    metadata={k:str(v) if isinstance(v,torch.device) else v for k,v in vars(opt).items()}
    metadata.update(checkpoint_sha256=checksum,annotation_sha256=hashlib.sha256(annotation.read_bytes()).hexdigest(),
                    selected_frames={sequences[k]:[im['frame_id'] for im in v] for k,v in images.items()})
    (output/'evaluation_config.json').write_text(json.dumps(metadata,indent=2))
    if accs:
        summary=Evaluator.get_summary(accs,names,mm.metrics.motchallenge_metrics)
        summary.to_csv(output/'clear_summary.csv')
        print(mm.io.render_summary(summary,formatters=mm.metrics.create().formatters,namemap=mm.io.motchallenge_metric_names))
    print('Tracks written to',output)
    print('Use the TrackEval wrapper for canonical HOTA/CLEAR/Identity metrics.')


if __name__ == '__main__':
    main(opts().init())
