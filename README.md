# amodalAPI

This is the PythonAPI to access the dataset proposed in [Semantic Amodal Segmentation](https://arxiv.org/abs/1509.01329). This API is largely built on [COCO API](https://github.com/pdollar/coco).

## setup and download annotation json files

1. install coco PythonAPI
  * ```cd PythonAPI; python setup.py build_ext install; cd ..```

2. create soft link of the coco images:
  * ```ln -s /your/coco/images ./images```

3. dowload amodal annotation files:
  * ```wget https://www.dropbox.com/s/4znu1sbf67c8c96/annotations.tar.gz```
  * ```tar xvf annotations.tar.gz```
  


## view the amodal annotation

  1. To see the annotation format/structure and some useful API, please run the ipython notebook: 
  https://github.com/Wakeupbuddy/amodalAPI/blob/master/PythonAPI/myAmodalDemo.ipynb
  
## evaluate

1. dowload an example json output on coco val set:

  * ```cd ./amodalAPI```
  * ```wget https://www.dropbox.com/s/5ermfba905l0dsb/exampleOutput.tar.gz```
  * ```tar xvf exampleOutput.tar.gz```
  
2. run the segmentation evaluation. 
  * ```bash eval.sh```
  
  It measure amodal segment proposal qualities using metric of average recall. Please see table 3a and section 5.1 in [the paper] (https://arxiv.org/abs/1509.01329).

  (The evaluation code need some cleanup. The depth ordering evaluation code will be released after the cleanup.)
  
## annotation tool

We also relase the web tool we used for annotation in another repo here: https://github.com/Wakeupbuddy/amodal-ui . It's a modified version from [OpenSurface](https://github.com/seanbell/opensurfaces).


