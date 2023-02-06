# ODIN: Out-of-Distribution Detector for Neural Networks

This is a fork of [facebookresearch/odin](https://github.com/facebookresearch/odin) written in a morden way with the power of [functorch](https://github.com/pytorch/functorch) and [TorchMetrics](https://github.com/Lightning-AI/metrics).

The method is described in the paper [Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks](https://arxiv.org/abs/1706.02690) by S. Liang, [Yixuan Li](www.yixuanli.net) and [R. Srikant](https://sites.google.com/a/illinois.edu/srikant/).

## Running the code

### Dependencies

Tested on:

* `pytorch`==1.13.1
* `torchmetrics`==0.11.1
* `tqdm`, `matplotlib`, ...

### Downloading
In the **root** of the repository, run
```bash
sh download.sh
```

#### Out-of-Distribtion Datasets
[facebookresearch/odin](https://github.com/facebookresearch/odin) provide download links of five out-of-distributin datasets:

* **[Tiny-ImageNet (crop)](https://www.dropbox.com/s/avgm2u562itwpkl/Imagenet.tar.gz)**
* **[Tiny-ImageNet (resize)](https://www.dropbox.com/s/kp3my3412u5k9rl/Imagenet_resize.tar.gz)**
* **[LSUN (crop)](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)**
* **[LSUN (resize)](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)**
* **[iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)**

#### Neural Network Models
We can use any pytorch model.

[facebookresearch/odin](https://github.com/facebookresearch/odin) provide download links of four pre-trained models.

* [DenseNet-BC trained on CIFAR-10](https://www.dropbox.com/s/wr4kjintq1tmorr/densenet10.pth.tar.gz)
* [DenseNet-BC trained on CIFAR-100](https://www.dropbox.com/s/vxuv11jjg8bw2v9/densenet100.pth.tar.gz)
* ~~[Wide ResNet trained on CIFAR-10](https://www.dropbox.com/s/uiye5nw0uj6ie53/wideresnet10.pth.tar.gz)~~
* ~~[Wide ResNet trained on CIFAR-100](https://www.dropbox.com/s/elfw7e3uofpydg5/wideresnet100.pth.tar.gz)~~

DenseNet-BC models prints some warnings, but they work.
Wide ResNet models need 3 GPUs and older version of pytorch (NOT tested).

### Running
See `ODIN.ipynb`.