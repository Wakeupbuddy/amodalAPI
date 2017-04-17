# Semantic Amodal Segmentation dataset and API

This is the Python API code for the amodal segmentation dataset proposed in [Semantic Amodal Segmentation](https://arxiv.org/abs/1509.01329) (CVPR 2017). This API code is built on [COCO API](https://github.com/pdollar/coco).

## setup

1. git clone and compile:
  * ```git clone https://github.com/wakeupbuddy/amodalAPI```
  * ```cd PythonAPI; python setup.py build_ext install; cd ..```

2. create soft link for coco/bsds images:
  * ```ln -s /your/coco/images ./images```
  * ```ln -s /your/bsds/images ./bsds_images```
3. dowload [annotation files](https://drive.google.com/open?id=0B8e3LNo7STslZURoTzhhMFpCelE) and untar.

## notebook demo

  1. To see the annotation format/structure and some useful APIs, please run the [ipython notebook](https://github.com/Wakeupbuddy/amodalAPI/blob/master/PythonAPI/myAmodalDemo.ipynb).

## evaluate

1. dowload the [baseline amodalMask output](https://drive.google.com/open?id=0B8e3LNo7STslUGRFUVlQSnZRUVE) on coco val set and untar:
  
2. run the segmentation evaluation. 
  * ```bash eval.sh```
  
  It measures amodal segment proposal quality using average recall. Please see details in table 3a and section 5.1 from [the paper] (https://arxiv.org/abs/1509.01329).
  
## annotation tool

We also release the web tool we used for annotation in another repo [here](https://github.com/Wakeupbuddy/amodal-ui). It's modified based on [OpenSurface](https://github.com/seanbell/opensurfaces).

## citation

If you find this dataset useful to your research, please consider citing:
```
@inproceedings{zhu2017semantic,
    Author = {Zhu, Yan and Tian, Yuandong and Mexatas, Dimitris and Doll{\'a}r, Piotr},
    Title = {Semantic Amodal Segmentation},
    Booktitle = {Conference on Computer Vision and Pattern Recognition ({CVPR})},
    Year = {2017}
}
```
