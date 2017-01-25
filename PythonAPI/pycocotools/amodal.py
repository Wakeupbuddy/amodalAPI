__author__ = 'yzhu'
__version__ = '0.1'
# Interface for accessing the Amodal dataset, most documentation are out of date right now..

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  segToMask  - Convert polygon segmentation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>segToMask, COCO>showAnns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import json
import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from skimage.draw import polygon
import urllib
import copy
import itertools
import mask
import os
from pycocotools.coco import COCO
import pdb
dd = pdb.set_trace

class Amodal(COCO):
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        COCO.__init__(self, annotation_file)

    def createIndex(self):
        # create index
        print 'creating index...'
        anns = {}
        imgToAnns = {}
        imgs = {}
        regions = []
        if 'annotations' in self.dataset:
            imgToAnns = {ann['image_id']: [] for ann in self.dataset['annotations']}
            anns =      {ann['id']:       [] for ann in self.dataset['annotations']}
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']] += [ann]
                anns[ann['id']] = ann
                for region in ann['regions']:
                    region['image_id'] = ann['image_id']
                    regions.append(region)

        if 'images' in self.dataset:
            imgs      = {im['id']: {} for im in self.dataset['images']}
            for img in self.dataset['images']:
                imgs[img['id']] = img
                #filenameToImgs[img['file_name']] = (img['id'], img['height'], img['width'])
                #filenameToImgs[img['file_name']] = img

        print 'index created!'

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.imgs = imgs
        self.regions = regions

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        print("todo")

    def getAmodalAnnIds(self, imgIds=[]):
        """
        Get amodal ann id that satisfy given fiter conditions.
        :param imgIds (int array): get anns for given imgs
        :return: ids (int array) : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]

        if len(imgIds) == 0:
            anns = self.dataset['annotations']
        else:
            lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
            anns = list(itertools.chain.from_iterable(lists))
        ids = [ann['id'] for ann in anns]

        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def showAmodalAnns(self, anns):
        """
        Display a set of amodal Ann object.
        :param anns: a dict object
        return: None
        """
        if type(anns) == list:
            print("anns cannot be a list! Should be a dict")
            return 0
        ax = plt.gca()
        polygons = []
        lines = []
        color = []
        #print("total num: ")
        #print(anns['size'])
        for ann in reversed(anns['regions']):
            c = np.random.random((1, 3)).tolist()[0]
            if type(ann['segmentation']) == list:
                # polygon
                seg = ann['segmentation']
                poly = np.array(seg).reshape((len(seg)/2, 2))
                polygons.append(Polygon(poly, True, alpha=0.2))
                color.append(c)
            else:
                #mask fixme
                print("todo")
        p = PatchCollection(polygons, facecolors=color, edgecolors=(0,0,0,1), linewidths=3, alpha=0.8)
        ax.add_collection(p)

    def showEdgeMap(self, anns):
        """
        Show edge map for an annontation
        :param anns: a dict object
        return: None
        """
        if type(anns) == list:
            print("anns cannot be a list! Should be a dict")
            return 0
        ax = plt.gca()
        polygons = []
        lines = []
        color = []
        #print("total num: ")
        #print(anns['size'])
        for ann in reversed(anns['regions']):
            #c = np.random.random((1, 3)).tolist()[0]
            c = np.zeros([1, 3]).tolist()[0]
            if type(ann['segmentation']) == list:
                # polygon
                seg = ann['segmentation']
                poly = np.array(seg).reshape((len(seg)/2, 2))
                polygons.append(Polygon(poly, True, alpha=0.2))
                color.append(c)
            else:
                #mask fixme
                print("todo")
        p = PatchCollection(polygons, facecolors=color, edgecolors=(1,1,1,1), linewidths=1, alpha=1)
        ax.add_collection(p)

    def showMask(self, M, ax, c = [0, 1, 0]):
        m = mask.decode([M])
        img = np.ones( (m.shape[0], m.shape[1], 3) )
#        color_mask = np.array([1.0, 1.0, 1.0])/255
        #c = [0.0, 1.0, 0.0] # green
        
        # get boundary quickly
        B = np.zeros( (m.shape[0], m.shape[1]) )
        for aa in range(m.shape[0]-1):
            for bb in range(m.shape[1]-1):
                #kk = aa*m.shape[1]+bb
                if m[aa,bb] != m[aa,bb+1]:
                    B[aa,bb], B[aa,bb+1] = 1,1
                if m[aa,bb] != m[aa+1,bb]:
                    B[aa,bb], B[aa+1,bb] = 1,1
                if m[aa,bb] != m[aa+1,bb+1]:
                    B[aa,bb], B[aa+1,bb+1] = 1,1
                    
        for i in range(3):
            img[:, :, i] = c[i]
            ax.imshow(np.dstack( (img, B*1) ))
            ax.imshow(np.dstack( (img, m*0.3) ))

        return
 
    def showAmodalInstance(self, anns, k=-1):
        """
        Display k-th instance only: print segmentation first, then print invisible_mask
        anns: a single annotation
        k: the depth order of anns, 1-index. If k = -1, just visulize input
        """
        ax = plt.gca()
        c = np.random.random((1,3)).tolist()[0]
        c = [0.0, 1.0, 0.0] # green
        
        if k < 0:
           self.showMask(anns['segmentation'], ax)
           return

        if type(anns) == list:
            print("ann cannot be a list! Should be a dict")
            return 0
        ann = anns['regions'][k-1]
        polygons = []
        color = []
        # draw whole mask
        if type(ann['segmentation']) == list:
            # polygon
            seg = ann['segmentation']
            poly = np.array(seg).reshape((len(seg)/2, 2))
            polygons.append(Polygon(poly, True, alpha=0.2))
            color.append(c)
            p = PatchCollection(polygons, facecolors=color, edgecolors=(1,1,1,1), linewidths=3, alpha=0.2)
            ax.add_collection(p)
        else:
            self.showMask(ann['segmentation'], ax)
        
        # draw invisible_mask
        if 'invisible_mask' in ann:
            self.showMask(ann['invisible_mask'], ax, [1, 0, 0])

    def showModalInstance(self, anns, k):
        """
        Display k-th instance: print its visible mask
        anns: a single annotation
        k: the depth order of anns, 1-index
        """
        if type(anns) == list:
            print("ann cannot be a list! Should be a dict")
            return 0
        ax = plt.gca()
        c = np.random.random((1,3)).tolist()[0]
        c = [0.0, 1.0, 0.0] # green
        ann = anns['regions'][k-1]
        polygons = []
        color = []
        # draw whole mask
        if 'visible_mask' in ann:
            mm = mask.decode([ann['visible_mask']])
            img = np.ones( (mm.shape[0], mm.shape[1], 3) )
            color_mask = c
            for i in range(3):
                img[:, :, i] = color_mask[i]
            ax.imshow(np.dstack( (img, mm*0.6) ))
        else:
            if type(ann['segmentation']) == list:
                # polygon
                seg = ann['segmentation']
                poly = np.array(seg).reshape((len(seg)/2, 2))
                polygons.append(Polygon(poly, True, alpha=0.2))
                color.append(c)
            else:
                #mask fixme
                mm = mask.decode([ann['segmentation']])
                img = np.ones( (mm.shape[0], mm.shape[1], 3) )
                color_mask = c
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack( (img, mm*0.6) ))
            
            p = PatchCollection(polygons, facecolors=color, edgecolors=(0,0,0,1), linewidths=3, alpha=0.4)
            ax.add_collection(p)

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = Amodal()
        res.dataset['images'] = [img for img in self.dataset['images']]
        # res.dataset['info'] = copy.deepcopy(self.dataset['info'])
        # res.dataset['licenses'] = copy.deepcopy(self.dataset['licenses'])

        print 'Loading and preparing results...     '
        tic = time.time()
        anns    = json.load(open(resFile))
        assert type(anns) == list, 'results in not an array of objects'
        annsImgIds = [ann['image_id'] for ann in anns]
        
        assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
               'Results do not correspond to current coco set'
       
        print 'DONE (t=%0.2fs)'%(time.time()- tic)
        res.dataset['annotations'] = anns
        res.createIndex()
        return res
    def download( self, tarDir = None, imgIds = [] ):
        print("todo")
