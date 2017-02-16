from pycocotools.amodal import Amodal
from pycocotools.amodaleval import AmodalEval
import numpy as np
import json
import os,sys
import glob
import pdb
dd = pdb.set_trace

import argparse
def parse_args(args):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--useAmodalGT', type=int, default=1, help='1 use amodal GT; 0 use visible GT.')
    parser.add_argument('--useAmodalDT', type=int, default=0, help='1 use amodal DT; 0 use modal DT.')
    parser.add_argument('--onlyThings', type=int, default=1, help='2: stuff only; 1: things only; 0: both')
    parser.add_argument('--occRange', type=str, default='all', help='support slight, medium, severe and all')
    parser.add_argument('--maxProp', type=int, default=100, help='support slight, medium, severe and all')
    params = parser.parse_args(args)
    return params

def createAmodalRegion(ann, id):
    region = {}
    region['id'] = id #used for gt/dt matching
    region['segmentation'] = ann['segmentation']
    region['score'] = ann['score']
    region['isStuff'] = 0  # default things
    if 'foreground_ness' in ann:
        region['foreground_ness'] = ann['foreground_ness']
    if 'invisibleMask' in ann:
        region['invisible_mask'] = ann['invisibleMask']
    if 'amodalMask' in ann:
        region['amodal_mask'] = ann['amodalMask']
    return region

def createAmodalAnn(image_id, ann_id):
    ann = {}
    ann['id'] = ann_id
    ann['category_id'] = 1 # fake label
    ann['image_id'] = image_id
    ann['regions']  =[]
    return ann

# filter coco dt, according to amodal gt imgId
# coco dt is formated as instance. This function transform data into image_id view
def filterDtFile(resFiles, amodalGtImgIds):
    amodalDt = {}
    id = 0
    ann_id = 0
    for i, file in enumerate(resFiles):
        print "processing json %d in total %d" %(i+1, len(resFiles))
        anns = json.load(open(file))
        for ann in anns:
            image_id = ann['image_id']
            if image_id in amodalGtImgIds:
                id = id + 1
                if image_id not in amodalDt:
                    amodalDt[image_id] = createAmodalAnn(image_id, ann_id)
                    ann_id = ann_id + 1
                region = createAmodalRegion(ann, id)
                amodalDt[image_id]['regions'].append(region)
    res = []
    for image_id, ann in amodalDt.iteritems():
        res.append(ann)
    return res

def evalWrapper(amodalDt, amodalGt, useAmodalGT, useAmodalDT, onlyThings, occRange, maxProp):
    annType = 'segm'
    imgIds=sorted(amodalGt.getImgIds())

    amodalEval = AmodalEval(amodalGt,amodalDt)
    amodalEval.params.imgIds  = imgIds
    amodalEval.params.useSegm = (annType == 'segm')
    if maxProp == 1000:
        amodalEval.params.maxDets = [1,10,100,1000]
    amodalEval.params.useAmodalGT = useAmodalGT # 1: use amodal GT; 0 use visible GT
    amodalEval.params.useAmodalDT = useAmodalDT # 1: use amodal DT; 0 use modal DT
    amodalEval.params.onlyThings = onlyThings # 2: stuff only; 1: things only; 0: both, 3: only select stuff vs things in constraints

    if occRange == 'all':
        amodalEval.params.occRng = [0, 1]
    elif occRange == 'none':
        amodalEval.params.occRng = [0, 0.00001] # only non occlusion regions
    elif occRange == 'partial':
        amodalEval.params.occRng = [0.00001, 0.25] # exclude non-occluding regions
    elif occRange == 'heavy':
        amodalEval.params.occRng = [0.25, 1]
    else:
        raise Exception('invalid occRange')

    amodalEval.evaluate()
    #annotatedGT = amodalEval.exportDtFile(matchedDtFile)
    amodalEval.accumulate()
    stats = amodalEval.summarize()
    return amodalEval.stats

if __name__ == "__main__":
    args=parse_args(sys.argv[1:])
    stats = evalWrapper(**vars(args))
