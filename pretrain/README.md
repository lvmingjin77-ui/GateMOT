ImageNet weights used to initialize the DLA encoder. They are not tracking checkpoints.

```bash
wget -P pretrain http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth
wget -P pretrain http://dl.yf.io/dla/models/imagenet/dla60-24839fc4.pth
```

DLA-34 is the default backbone. SportsMOT uses DLA-60.
