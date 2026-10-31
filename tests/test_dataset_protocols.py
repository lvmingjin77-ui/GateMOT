"""Dataset protocol checks: FRCNN-only MOT17 and ceil temporal halves."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/lib'), str(ROOT / 'src/tools')]
from mot_format import convert


def sequence(root, split, name, count, with_gt=True):
    directory = root / split / name
    (directory / 'img1').mkdir(parents=True)
    for frame in range(1, count + 1):
        cv2.imwrite(str(directory / 'img1' / f'{frame:06d}.jpg'),
                    np.full((24, 32, 3), frame, dtype=np.uint8))
    if with_gt:
        (directory / 'gt').mkdir()
        rows = [f'{frame},10,4,4,8,8,1,1,1\n' for frame in range(1, count + 1)]
        # Include identities seen only in the first or last frame.
        rows += ['1,20,12,4,8,8,1,1,1\n', f'{count},30,20,4,8,8,1,1,1\n']
        (directory / 'gt/gt.txt').write_text(''.join(rows))
    return directory


def manifest(root, split):
    return json.loads((root / 'annotations' / f'{split}.json').read_text())


def frames_by_sequence(data):
    names = {video['id']: video['file_name'] for video in data['videos']}
    return {name: [image['frame_id'] for image in data['images'] if image['video_id'] == video]
            for video, name in names.items()}


def identities_by_source(data):
    images = {image['id']: image['file_name'].split('/')[0] for image in data['images']}
    result = {}
    for ann in data['annotations']:
        if ann['category_id'] == 1:
            key = (images[ann['image_id']], ann['source_track_id'])
            result.setdefault(key, set()).add(ann['track_id'])
    return result


class DatasetProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='gatemot_protocol_')
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_converter(self, dataset, root, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / 'src/tools' / f'{dataset}_to_coco.py'),
             '--data-root', str(root), *args],
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            check=True, capture_output=True, text=True)

    def assert_links_stay_in_split(self, data):
        images = {image['id']: image for image in data['images']}
        for video in data['videos']:
            frames = sorted((image for image in images.values() if image['video_id'] == video['id']),
                            key=lambda image: image['frame_id'])
            self.assertEqual(frames[0]['prev_image_id'], -1)
            self.assertEqual(frames[-1]['next_image_id'], -1)
            for index, image in enumerate(frames):
                self.assertEqual(image['prev_image_id'], frames[index - 1]['id'] if index else -1)
                self.assertEqual(image['next_image_id'], frames[index + 1]['id'] if index + 1 < len(frames) else -1)
                self.assertEqual(int(Path(image['file_name']).stem), image['frame_id'])
        self.assertTrue(all(ann['image_id'] in images for ann in data['annotations']))

    def test_mot17_entry_point_exports_only_frcnn_for_every_split(self):
        for split in ('train', 'test'):
            for number, count in (('02', 5), ('04', 4)):
                for detector in ('DPM', 'FRCNN', 'SDP'):
                    directory = sequence(self.root, split, f'MOT17-{number}-{detector}', count,
                                         with_gt=split == 'train')
                    if detector != 'FRCNN':
                        # Reading excluded detections' images or GT would fail.
                        (directory / 'img1/000001.jpg').write_text('excluded detector')
                        if split == 'train':
                            (directory / 'gt/gt.txt').write_text('excluded detector')
            for name in ('OTHER', 'MOT17-05-frcnn', 'MOT17-05-FRCNN-backup'):
                directory = sequence(self.root, split, name, 1, with_gt=False)
                (directory / 'img1/000001.jpg').write_text('excluded directory')
        self.run_converter('mot17', self.root, '--splits', 'train', 'test', 'halftrain', 'halfval')
        expected = {
            'train': {'MOT17-02-FRCNN': [1, 2, 3, 4, 5], 'MOT17-04-FRCNN': [1, 2, 3, 4]},
            'test': {'MOT17-02-FRCNN': [1, 2, 3, 4, 5], 'MOT17-04-FRCNN': [1, 2, 3, 4]},
            'halftrain': {'MOT17-02-FRCNN': [1, 2, 3], 'MOT17-04-FRCNN': [1, 2]},
            'halfval': {'MOT17-02-FRCNN': [4, 5], 'MOT17-04-FRCNN': [3, 4]},
        }
        for split, frame_map in expected.items():
            data = manifest(self.root, split)
            self.assertEqual(frames_by_sequence(data), frame_map)
            self.assertEqual(data['conversion']['sequence_policy'], 'name-endswith-FRCNN')
            self.assert_links_stay_in_split(data)
        self.assertEqual(manifest(self.root, 'test')['annotations'], [])

    def test_all_datasets_use_ceil_halves_with_stable_sequence_ids(self):
        for dataset in ('bee', 'mot17', 'mot20', 'sportsmot'):
            with self.subTest(dataset=dataset):
                root = self.root / dataset
                lengths = {'A_SINGLE': 1, 'B_EVEN': 4, 'C_ODD': 5}
                if dataset == 'mot17':
                    lengths = {name + '-FRCNN': count for name, count in lengths.items()}
                for name, count in lengths.items():
                    sequence(root, 'train', name, count)
                convert(dataset, root, ['halftrain', 'halfval', 'train'])
                data = {split: manifest(root, split) for split in ('train', 'halftrain', 'halfval')}
                paths = {split: {image['file_name'] for image in item['images']} for split, item in data.items()}
                self.assertFalse(paths['halftrain'] & paths['halfval'])
                self.assertEqual(paths['halftrain'] | paths['halfval'], paths['train'])
                for split, item in data.items():
                    self.assertEqual(item['conversion']['version'], 4)
                    self.assertEqual(item['conversion']['half_split_policy'], 'ceil')
                    self.assertEqual(item['conversion']['half_train_frames'], 'ceil(N/2)')
                    self.assertEqual(item['conversion']['identity_policy'], 'selected-split-sequence-and-source-ID')
                    self.assert_links_stay_in_split(item)
                    actual = frames_by_sequence(item)
                    for name, count in lengths.items():
                        boundary = (count + 1) // 2
                        expected = list(range(1, count + 1)) if split == 'train' else list(
                            range(1, boundary + 1) if split == 'halftrain' else range(boundary + 1, count + 1))
                        self.assertEqual(actual.get(name, []), expected)
                    identity_map = identities_by_source(item)
                    self.assertTrue(all(len(labels) == 1 for labels in identity_map.values()))
                    labels = {next(iter(values)) for values in identity_map.values()}
                    self.assertEqual(labels, set(range(1, len(identity_map) + 1)))

    def test_validation_only_identities_do_not_change_training_manifest_or_nid(self):
        from dataset.datasets.custom_dataset import CustomDataset

        for dataset in ('bee', 'mot17', 'mot20', 'sportsmot'):
            with self.subTest(dataset=dataset):
                root = self.root / dataset
                names = ['A-FRCNN', 'B-FRCNN'] if dataset == 'mot17' else ['A', 'B']
                directories = [sequence(root, 'train', name, 5) for name in names]
                convert(dataset, root, ['halftrain'])
                annotation = root / 'annotations/halftrain.json'
                before = annotation.read_bytes()

                def load_identity_count():
                    opt = SimpleNamespace(
                        K=10, num_classes=1, input_h=32, input_w=32,
                        custom_dataset_img_path=str(root / 'train'),
                        custom_dataset_ann_path=str(annotation),
                        local_rank=1, tracking=True, reid=True)
                    CustomDataset(opt, 'train')
                    return opt.nid

                nid = load_identity_count()
                self.assertEqual(nid, 4)
                for directory in directories:
                    gt = directory / 'gt/gt.txt'
                    rows = gt.read_text().splitlines()
                    changed = []
                    for row in rows:
                        fields = row.split(',')
                        if int(fields[0]) > 3:
                            fields[1] = str(1000 + int(fields[1]))
                        changed.append(','.join(fields))
                    changed.extend(['4,1,4,4,8,8,1,1,1', '5,2,4,4,8,8,1,1,1'])
                    gt.write_text('\n'.join(changed) + '\n')
                convert(dataset, root, ['halftrain', 'halfval'])
                self.assertEqual(annotation.read_bytes(), before)
                self.assertEqual(load_identity_count(), nid)

    def test_independent_entry_points_keep_full_official_splits_and_test_needs_no_gt(self):
        for dataset in ('bee', 'mot20', 'sportsmot'):
            with self.subTest(dataset=dataset):
                root = self.root / dataset
                sequence(root, 'train', 'TRAIN', 5)
                sequence(root, 'test', 'TEST', 5, with_gt=False)
                if dataset == 'sportsmot':
                    sequence(root, 'val', 'VAL', 5)
                self.run_converter(dataset, root)
                expected = {'train', 'val', 'test'} if dataset == 'sportsmot' else {'train', 'test'}
                self.assertEqual({path.stem for path in (root / 'annotations').glob('*.json')}, expected)
                for split in expected:
                    data = manifest(root, split)
                    self.assertEqual([image['frame_id'] for image in data['images']], [1, 2, 3, 4, 5])
                    self.assertEqual(data['conversion']['source_split'], split)
                self.assertEqual(manifest(root, 'test')['annotations'], [])

    def test_mot_training_exports_only_marked_pedestrians(self):
        for dataset in ('mot17', 'mot20'):
            with self.subTest(dataset=dataset):
                root = self.root / dataset
                name = 'MOT17-02-FRCNN' if dataset == 'mot17' else 'MOT20-01'
                directory = sequence(root, 'train', name, 2)
                with (directory / 'gt/gt.txt').open('a') as handle:
                    for label in (0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
                        handle.write(f'1,{100 + label},4,4,8,8,1,{label},1\n')
                    handle.write('1,200,4,4,8,8,0,1,1\n')
                convert(dataset, root, ['train'])
                data = manifest(root, 'train')
                self.assertEqual(len(data['annotations']), 4)
                self.assertEqual({ann['category_id'] for ann in data['annotations']}, {1})
                self.assertEqual({ann['source_track_id'] for ann in data['annotations']}, {10, 20, 30})
                self.assertTrue(all(ann['conf'] > 0 and ann['track_id'] > 0 for ann in data['annotations']))


if __name__ == '__main__':
    unittest.main()
