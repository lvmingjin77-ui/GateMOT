"""Evaluate a prepared bundle with the upstream TrackEval Python API.

Restore NumPy scalar aliases used by the pinned upstream revision in this
process only. No upstream files or metric definitions are modified.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trackeval', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--preproc', choices=['True', 'False'], default='False')
    args = parser.parse_args()
    root = Path(args.trackeval).resolve()
    bundle = Path(args.bundle).resolve()
    if not (root / 'trackeval/__init__.py').is_file():
        raise FileNotFoundError('TrackEval source not found at {}'.format(root))
    for name, scalar in (('float', float), ('int', int), ('bool', bool)):
        if name not in np.__dict__:
            setattr(np, name, scalar)
    sys.path.insert(0, str(root))
    import trackeval
    evaluator_config = {'USE_PARALLEL': False, 'PLOT_CURVES': False,
                        'LOG_ON_ERROR': str(bundle / 'error_log.txt')}
    dataset_config = dict(GT_FOLDER=str(bundle/'gt'),
        TRACKERS_FOLDER=str(bundle/'trackers'), TRACKERS_TO_EVAL=[''],
        TRACKER_SUB_FOLDER='', SPLIT_TO_EVAL='train', SKIP_SPLIT_FOL=True,
        SEQMAP_FILE=str(bundle/'seqmap.txt'), DO_PREPROC=args.preproc=='True',
        GT_LOC_FORMAT='{gt_folder}/{seq}/gt/gt.txt')
    provenance = dict(evaluator_config=evaluator_config, dataset_config=dataset_config,
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (root/'trackeval').rglob('*.py')})
    (bundle/'evaluator_config.json').write_text(json.dumps(provenance,indent=2))
    evaluator = trackeval.Evaluator(evaluator_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator.evaluate([dataset], metrics)


if __name__ == '__main__':
    main()
