#!/usr/bin/env python
"""
验证生成的标注文件是否正确
检查frame_id是否使用原始帧号
"""
import json
import os

DATA_PATH = os.environ.get('DATA_ROOT', '')
if not DATA_PATH:
    raise SystemExit('Set DATA_ROOT to the MOT17 root.')
ANN_PATH = os.path.join(DATA_PATH, 'annotations')

def check_annotation(split):
    print(f"\n{'='*60}")
    print(f"检查 {split}.json")
    print('='*60)
    
    ann_file = os.path.join(ANN_PATH, f'{split}.json')
    if not os.path.exists(ann_file):
        print(f"❌ 文件不存在: {ann_file}")
        return False
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print(f"✓ 视频数量: {len(data['videos'])}")
    print(f"✓ 图像数量: {len(data['images'])}")
    print(f"✓ 标注数量: {len(data['annotations'])}")
    
    # 检查每个视频的frame_id
    videos = {v['id']: v['file_name'] for v in data['videos']}
    
    for video_id, video_name in videos.items():
        imgs = [img for img in data['images'] if img['video_id'] == video_id]
        imgs = sorted(imgs, key=lambda x: x['frame_id'])
        
        if len(imgs) > 0:
            first_frame = imgs[0]['frame_id']
            last_frame = imgs[-1]['frame_id']
            
            # 检查是否连续
            frame_ids = [img['frame_id'] for img in imgs]
            
            print(f"\n  视频: {video_name}")
            print(f"    - 图像数: {len(imgs)}")
            print(f"    - frame_id 范围: {first_frame} ~ {last_frame}")
            
            # 关键检查：halfval的frame_id应该从中间开始，而不是从1开始
            if split == 'halfval' and first_frame == 1:
                print(f"    ⚠️  警告: halfval的frame_id从1开始（应该从中间开始）")
                print(f"    这可能导致test.py出错！")
                return False
            elif split == 'halfval':
                print(f"    ✅ halfval的frame_id正确（从{first_frame}开始）")
    
    return True

if __name__ == '__main__':
    splits = ['halftrain', 'halfval', 'train', 'test']
    
    print("MOT17标注文件验证")
    print("="*60)
    
    all_ok = True
    for split in splits:
        ok = check_annotation(split)
        if not ok:
            all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ 所有标注文件验证通过！")
    else:
        print("❌ 部分标注文件有问题，请重新生成")
    print("="*60)






