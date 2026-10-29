"""Prepare an exact frame-subset evaluation bundle without altering source GT."""
import argparse
import configparser
import json
from collections import defaultdict
from pathlib import Path


def prepare(images_root, annotations, tracks, output, dataset):
    images_root,tracks,output=map(Path,(images_root,tracks,output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose an empty evaluation bundle directory')
    data=json.loads(Path(annotations).read_text())
    names={v['id']:v['file_name'] for v in data['videos']}
    images=defaultdict(list)
    for image in data['images']:
        images[image['video_id']].append(image)
    output.mkdir(parents=True,exist_ok=True)
    sequence_names=[]
    for vid,items in images.items():
        name=names[vid]
        sequence_names.append(name)
        items.sort(key=lambda x:x['frame_id'])
        frame_map={im['frame_id']:i+1 for i,im in enumerate(items)}
        gt=images_root/name/'gt/gt.txt'
        prediction=tracks/(name+'.txt')
        if not gt.is_file() or not prediction.is_file():
            raise FileNotFoundError('GT and tracking txt required for '+name)
        for source,dest,is_gt in [(gt,output/'gt'/name/'gt/gt.txt',True),
                                   (prediction,output/'trackers'/(name+'.txt'),False)]:
            dest.parent.mkdir(parents=True,exist_ok=True)
            rows=[]
            for line in source.read_text().splitlines():
                fields=line.strip().split(',')
                if len(fields)<6:
                    continue
                frame=int(float(fields[0]))
                if frame not in frame_map:
                    continue
                fields[0]=str(frame_map[frame])
                if is_gt and dataset not in ('mot17','mot20'):
                    mark=fields[6] if len(fields)>6 else '1'
                    fields=fields[:6]+[mark,'1','1']
                rows.append(','.join(fields))
            dest.write_text('\n'.join(rows)+ ('\n' if rows else ''))
        config=configparser.ConfigParser()
        config['Sequence']=dict(name=name,imDir='img1',frameRate='30',seqLength=str(len(items)),
            imWidth=str(items[0]['width']),imHeight=str(items[0]['height']),imExt='.jpg')
        with (output/'gt'/name/'seqinfo.ini').open('w') as handle:
            config.write(handle)
    (output/'seqmap.txt').write_text('name\n'+'\n'.join(sequence_names)+'\n')
    (output/'subset.json').write_text(json.dumps({'source_annotations':str(annotations),'dataset':dataset,'sequences':sequence_names},indent=2))
    print(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-root',required=True)
    parser.add_argument('--annotations',required=True)
    parser.add_argument('--tracks',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--dataset',required=True,choices=['bee24','mot17','mot20','sportsmot'])
    prepare(**vars(parser.parse_args()))
