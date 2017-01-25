from pycocotools.amodal import Amodal
from pycocotools.amodaleval import AmodalEval
#from pycocotools.cocoeval import COCOeval
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
    parser.add_argument('--sortKey', type=str, default='oracle', help='support oracle, score, random now')
    parser.add_argument('--occRange', type=str, default='all', help='support slight, medium, severe and all')
    parser.add_argument('--maxProp', type=int, default=100, help='support slight, medium, severe and all')
    params = parser.parse_args(args)
    return params

def createAmodalRegion(ann, id):
    region = {}
    region['id'] = id # unique id, used for gt/dt matching.
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
    # front_ness is empty

def createAmodalAnn(image_id, ann_id):
    ann = {}
    ann['id'] = ann_id # no real usage for a ann level id..
    ann['category_id'] = 1 # fake label
    ann['image_id'] = image_id
    ann['regions']  =[]
    return ann

# because amodal gt is sparser than coco gt, 
# we need to filter coco dt, according to amodal gt imgId
# also, coco dt is formated as instance. This function also
# transform data into image_id view
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
                # suppose a modified algorithm can output invisible_mask and foreground_ness, then use following line
                #region = createAmodalRegion(ann['segmentation'], ann['score'], ann['invisible_mask'], ann['foreground_ness'])
                amodalDt[image_id]['regions'].append(region)
    res = []
    for image_id, ann in amodalDt.iteritems():
        #saveName = '../results/sm1k_val/sm1k_val_'+str(ann['image_id'])+'.json'
        #with open(saveName, 'wb') as output:
        #    json.dump(ann, output)
        res.append(ann)
    return res

# sanity check code
def filterGt(amodalGt):
    cnt = 1
    for id, ann in amodalGt.anns.iteritems():
        for region in ann['regions']:
            region['score'] = 1.0/(1+cnt)
            region['id'] = cnt
            cnt = cnt + 1
    return amodalGt 

def evalWrapper(amodalDt, amodalGt, useAmodalGT, useAmodalDT, onlyThings, sortKey, occRange, maxProp):
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

    amodalEval.params.sortKey = sortKey # support 'oracle', 'score', 'random' now
    if occRange == 'all':
        amodalEval.params.occRng = [0, 1]
    elif occRange == 'none':
        amodalEval.params.occRng = [0, 0.00001] # only non occlusion regions
    elif occRange == 'partial':
        amodalEval.params.occRng = [0.00001, 0.25] # exclude non-occluding regions
    #elif occRange == 'medium':
    #    amodalEval.params.occRng = [0.2, 0.5]
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
