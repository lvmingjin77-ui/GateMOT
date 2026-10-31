"""MOT-layout conversion with stable, sequence-scoped identity labels."""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def _ground_truth_rows(dataset, sequence, frame_ids):
    gt = sequence / 'gt/gt.txt'
    if not gt.is_file():
        raise FileNotFoundError(gt)
    rows = np.loadtxt(str(gt), delimiter=',', ndmin=2) if gt.stat().st_size else np.empty((0, 9))
    selected = []
    for row in rows:
        if len(row) < 6:
            raise ValueError('GT rows require frame, ID, x, y, width and height: {}'.format(gt))
        if int(row[0]) not in frame_ids or row[4] <= 0 or row[5] <= 0:
            continue
        if len(row) > 6 and row[6] <= 0:
            continue
        if dataset in ('mot17', 'mot20'):
            if len(row) < 8:
                raise ValueError('MOT17/20 GT requires class and mark fields')
            # The custom training loader has one category: marked pedestrians.
            if int(row[7]) != 1:
                continue
        selected.append(row)
    return selected


def convert(dataset, root, splits=None):
    root = Path(root)
    output = root / 'annotations'
    output.mkdir(parents=True, exist_ok=True)
    if splits is None:
        splits = (['train', 'val', 'test'] if dataset == 'sportsmot'
                  else ['train', 'test'])
    for split in splits:
        source = 'train' if split.startswith('half') else split
        directory = root / source
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        data = dict(images=[], annotations=[], videos=[], categories=[dict(id=1, name='bee' if dataset == 'bee' else 'person')])
        identity_map = {}
        for sequence in sorted(directory.iterdir()):
            # MOT17's detector variants repeat each video. Use exactly FRCNN.
            if dataset == 'mot17' and not sequence.name.endswith('-FRCNN'):
                continue
            image_dir = sequence / 'img1'
            if not image_dir.is_dir():
                continue
            names = sorted((p for p in image_dir.iterdir() if p.suffix.lower() == '.jpg'), key=lambda p: int(p.stem))
            if not names:
                continue
            boundary = (len(names) + 1) // 2
            source_frame_count = len(names)
            if split == 'halftrain':
                names = names[:boundary]
            elif split == 'halfval':
                names = names[boundary:]
            if not names:
                continue
            # Training labels depend only on this split's valid annotations.
            # source_track_id preserves the original identity across splits.
            rows = [] if split == 'test' else _ground_truth_rows(
                dataset, sequence, {int(path.stem) for path in names})
            for identity in sorted({int(row[1]) for row in rows}):
                identity_map[(sequence.name, identity)] = len(identity_map) + 1
            video_id = len(data['videos']) + 1
            data['videos'].append(dict(id=video_id, file_name=sequence.name,
                source_frame_count=source_frame_count, selected_frame_count=len(names),
                first_frame_id=int(names[0].stem), last_frame_id=int(names[-1].stem)))
            frame_map = {}
            base = len(data['images'])
            for index, path in enumerate(names):
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError('Unreadable image {}'.format(path))
                h, w = image.shape[:2]
                image_id = base + index + 1
                frame_id = int(path.stem)
                frame_map[frame_id] = image_id
                data['images'].append(dict(id=image_id, video_id=video_id,
                    file_name='{}/img1/{}'.format(sequence.name, path.name),
                    frame_id=frame_id, height=h, width=w,
                    prev_image_id=image_id-1 if index else -1,
                    next_image_id=image_id+1 if index+1 < len(names) else -1))
            if split == 'test':
                continue
            for row in rows:
                frame, identity = int(row[0]), int(row[1])
                if frame not in frame_map:
                    continue
                key = (sequence.name, identity)
                data['annotations'].append(dict(id=len(data['annotations'])+1,
                    image_id=frame_map[frame], category_id=1,
                    track_id=identity_map[key],
                    source_track_id=identity, bbox=row[2:6].tolist(),
                    conf=float(row[6]) if len(row)>6 else 1.0,
                    iscrowd=0, area=float(row[4]*row[5])))
        data['conversion'] = dict(dataset=dataset, source_split=source, split=split,
                                  identity_policy='selected-split-sequence-and-source-ID',
                                  half_split_policy='ceil', half_train_frames='ceil(N/2)',
                                  sequence_policy=('name-endswith-FRCNN' if dataset == 'mot17'
                                                   else 'all-image-sequences'), version=4)
        (output/(split+'.json')).write_text(json.dumps(data))
        identities = {ann['track_id'] for ann in data['annotations'] if ann['category_id'] == 1}
        print('{}: {} images, {} annotations, {} identities'.format(split,len(data['images']),len(data['annotations']),len(identities)))


def main(dataset):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=os.environ.get('DATA_ROOT'))
    parser.add_argument('--splits', nargs='+',
                        choices=['train', 'val', 'test', 'halftrain', 'halfval'])
    args = parser.parse_args()
    if not args.data_root:
        parser.error('Set DATA_ROOT or --data-root')
    convert(dataset, args.data_root, args.splits)
