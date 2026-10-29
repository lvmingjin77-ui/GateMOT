"""MOT-layout conversion with stable, sequence-scoped identity labels."""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def convert(dataset, root, splits=None):
    root = Path(root)
    output = root / 'annotations'
    output.mkdir(parents=True, exist_ok=True)
    if splits is None:
        splits = (['train', 'val', 'test'] if dataset in ('sportsmot', 'dancetrack')
                  else ['halftrain', 'halfval', 'train', 'test'])
    for split in splits:
        source = 'train' if split.startswith('half') else split
        directory = root / source
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        data = dict(images=[], annotations=[], videos=[], categories=[dict(id=1, name='bee' if dataset == 'bee' else 'person')])
        identity_map = {}
        for sequence in sorted(directory.iterdir()):
            image_dir = sequence / 'img1'
            if not image_dir.is_dir():
                continue
            names = sorted((p for p in image_dir.iterdir() if p.suffix.lower() == '.jpg'), key=lambda p: int(p.stem))
            if not names:
                continue
            boundary = len(names) // 2
            # Preserve the existing CenterTrack-style MOT half boundary.
            if dataset in ('mot17', 'mot20'):
                boundary += 1
            if split == 'halftrain':
                names = names[:boundary]
            elif split == 'halfval':
                names = names[boundary:]
            if not names:
                continue
            video_id = len(data['videos']) + 1
            data['videos'].append(dict(id=video_id, file_name=sequence.name))
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
            gt = sequence / 'gt/gt.txt'
            if not gt.is_file():
                raise FileNotFoundError(gt)
            rows = np.loadtxt(str(gt), delimiter=',', ndmin=2) if gt.stat().st_size else np.empty((0,9))
            for row in rows:
                frame, identity = int(row[0]), int(row[1])
                if frame not in frame_map or len(row) < 6 or row[4] <= 0 or row[5] <= 0:
                    continue
                if len(row) > 6 and row[6] <= 0:
                    continue
                category = 1
                if dataset in ('mot17', 'mot20'):
                    if len(row) < 8:
                        raise ValueError('MOT17/20 GT requires class and mark fields')
                    label = int(row[7])
                    if label in (3,4,5,6,9,10,11):
                        continue
                    if label in (2,7,8,12):
                        category = -1
                key = (sequence.name, identity)
                if category == 1 and key not in identity_map:
                    identity_map[key] = len(identity_map) + 1
                data['annotations'].append(dict(id=len(data['annotations'])+1,
                    image_id=frame_map[frame], category_id=category,
                    track_id=identity_map[key] if category == 1 else -1,
                    source_track_id=identity, bbox=row[2:6].tolist(),
                    conf=float(row[6]) if len(row)>6 else 1.0,
                    iscrowd=0, area=float(row[4]*row[5])))
        data['conversion'] = dict(dataset=dataset, source_split=source, split=split,
                                  identity_policy='sequence-and-source-ID', version=2)
        (output/(split+'.json')).write_text(json.dumps(data))
        print('{}: {} images, {} annotations, {} identities'.format(split,len(data['images']),len(data['annotations']),len(identity_map)))


def main(dataset):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=os.environ.get('DATA_ROOT'))
    parser.add_argument('--splits', nargs='+')
    args = parser.parse_args()
    if not args.data_root:
        parser.error('Set DATA_ROOT or --data-root')
    convert(dataset, args.data_root, args.splits)
