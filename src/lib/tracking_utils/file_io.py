import os
from typing import Dict
import numpy as np

from loguru import logger


def write_results(filename, results_dict: Dict, data_type: str):
    if not filename:
        return
    path = os.path.dirname(filename)
    if not os.path.exists(path):
        os.makedirs(path)

    if data_type in ('mot', 'mcmot', 'lab'):
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian -1 -1 -10 {x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10 {score}\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, frame_data in results_dict.items():
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in frame_data:
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h, score=1.0)
                f.write(line)
    logger.info('Save results to {}'.format(filename))


def read_results(filename, data_type: str, is_gt=False, is_ignore=False):
    if data_type in ('mot', 'lab'):
        read_fun = read_mot_results
    else:
        raise ValueError('Unknown data type: {}'.format(data_type))

    return read_fun(filename, is_gt, is_ignore)


"""
labels={'ped', ...			% 1
'person_on_vhcl', ...	% 2
'car', ...				% 3
'bicycle', ...			% 4
'mbike', ...			% 5
'non_mot_vhcl', ...		% 6
'static_person', ...	% 7
'distractor', ...		% 8
'occluder', ...			% 9
'occluder_on_grnd', ...		%10
'occluder_full', ...		% 11
'reflection', ...		% 12
'crowd' ...			% 13
};
"""


def read_mot_results(filename, is_gt, is_ignore):
    valid_labels = {1}
    ignore_labels = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
    results_dict = dict()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            for line in f.readlines():
                # 自动检测分隔符：先尝试逗号（MOT格式），再尝试空格（CrowdTrack格式）
                if ',' in line:
                    linelist = line.split(',')
                else:
                    linelist = line.split()
                
                # CrowdTrack格式只有6个字段：frame_id, obj_id, x, y, w, h
                # MOT格式有10个字段：frame_id, obj_id, x, y, w, h, conf, class, visibility, ...
                if len(linelist) < 6:
                    continue
                
                fid = int(linelist[0])
                # CrowdTrack的frame_id从0开始，需要+1转换为MOT标准（从1开始）
                # MOT格式已经从1开始，不需要转换
                if ',' not in line:  # 空格分隔 = CrowdTrack格式
                    fid = fid + 1  # 0->1, 1->2, 2->3, ...
                else:  # 逗号分隔 = MOT格式
                    if fid < 1:
                        continue
                
                results_dict.setdefault(fid, list())
                
                if is_gt:
                    # 检查是否是完整的MOT格式（有class和mark字段）
                    if len(linelist) >= 8 and any(tag in filename for tag in ('MOT16-', 'MOT17-', 'MOT20-')):
                        label = int(float(linelist[7]))
                        mark = int(float(linelist[6]))
                        if mark == 0 or label not in valid_labels:
                            continue
                    # CrowdTrack格式：所有目标都有效
                    score = 1
                elif is_ignore:
                    if len(linelist) >= 8 and any(tag in filename for tag in ('MOT16-', 'MOT17-', 'MOT20-')):
                        label = int(float(linelist[7]))
                        if label not in ignore_labels:
                            continue
                    else:
                        continue
                    score = 1
                else:
                    # 对于tracking结果，如果有conf字段则使用，否则默认为1
                    if len(linelist) >= 7:
                        score = float(linelist[6])
                    else:
                        score = 1.0

                #if box_size > 7000:
                #if box_size <= 7000 or box_size >= 15000:
                #if box_size < 15000:
                    #continue

                tlwh = tuple(map(float, linelist[2:6]))
                target_id = int(linelist[1])

                results_dict[fid].append((tlwh, target_id, score))

    return results_dict


def unzip_objs(objs):
    if len(objs) > 0:
        tlwhs, ids, scores = zip(*objs)
    else:
        tlwhs, ids, scores = [], [], []
    tlwhs = np.asarray(tlwhs, dtype=float).reshape(-1, 4)

    return tlwhs, ids, scores