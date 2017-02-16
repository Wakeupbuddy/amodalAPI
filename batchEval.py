from __future__ import print_function
import os,sys,glob,json
import string
import random
import argparse
from pycocotools.amodal import Amodal
from PythonAPI.myPyAmodalEvalDemo import evalWrapper, filterDtFile

mystr = "{:10.4f}".format

class Metric(object):
    def __init__(self, name, maxProp):
        self.name = name
        self.ar1 = 0
        self.ar10 = 0
        self.ar100 = 0
        self.ar1000 = 0
        self.ap = 0
        self.ap_05 = 0
        self.ap_075 = 0
        self.ap_none = 0
        self.ap_partial = 0
        self.ap_heavy = 0
        self.ar_none = 0
        self.ar_partial = 0
        self.ar_heavy = 0
        self.maxProp = maxProp
    
    def summarize(self, outputFile = ''):
        if outputFile != '':
            fout = open(outputFile, 'a')
            myprint = lambda x: fout.write(x + '\n')
        else:
            myprint = lambda x: print(x)
        
        myprint("")
        myprint("== " + self.name)
        myprint("AR1: " + mystr(self.ar1))
        myprint("AR10: " + mystr(self.ar10))
        myprint("AR100: " + mystr(self.ar100))
        if self.maxProp == 1000:
            myprint("AR1000: " + mystr(self.ar1000))

        myprint("AR_none: " + mystr(self.ar_none))
        myprint("AR_partial: " + mystr(self.ar_partial))
        myprint("AR_heavy: " + mystr(self.ar_heavy))
       
        if outputFile != '':
            fout.close()

def singleEval(amodalDt, amodalGt, useAmodalGT=1, onlyThings=0, maxProp=1000):
    # eval on different occlusion levels
    name = ""
    if useAmodalGT == 1:
        name = name + "Amodal mask"
    elif useAmodalGT == 2:
        name = name + "visible mask"
    else:
        raise NotImplementedError

    if onlyThings == 1:
        name = name + ", things only"
    elif onlyThings == 2:
        name = name + ", stuff only"
    elif onlyThings == 0:
        name = name + ", both stuff and things"
    else:
        raise NotImplementedError

    metric = Metric(name, maxProp)
    
    useAmodalDT = 1
    occRange = 'all'
    stats = evalWrapper(amodalDt, amodalGt, useAmodalGT, useAmodalDT, onlyThings, occRange, maxProp)
    metric.ap = stats[0]
    metric.ap_05 = stats[1]
    metric.ap_075 = stats[2]
    metric.ar1 = stats[3]
    metric.ar10 = stats[4]
    metric.ar100 = stats[5]
    metric.ar1000 = stats[6]

    occRange = 'none'
    stats = evalWrapper(amodalDt, amodalGt,useAmodalGT, useAmodalDT, onlyThings, occRange, maxProp)
    metric.ap_none = stats[0]
    if maxProp == 100:
        metric.ar_none = stats[5]
    elif maxProp == 1000:
        metric.ar_none = stats[6]
    del stats

    occRange = 'partial'
    stats = evalWrapper(amodalDt, amodalGt,useAmodalGT, useAmodalDT, onlyThings, occRange, maxProp)
    metric.ap_partial = stats[0]
    if maxProp == 100:
        metric.ar_partial = stats[5]
    elif maxProp == 1000:
        metric.ar_partial = stats[6]
    del stats

    occRange = 'heavy'
    stats = evalWrapper(amodalDt, amodalGt, useAmodalGT, useAmodalDT, onlyThings, occRange, maxProp)
    metric.ap_heavy = stats[0]
    if maxProp == 100:
        metric.ar_heavy = stats[5]
    elif maxProp == 1000:
        metric.ar_heavy = stats[6]
    del stats
    
    return metric

def main(args):
    annFile = '%s/annotations/COCO_amodal_%s.json'%(args.dataDir,args.dataType)
    amodalGt=Amodal(annFile)
    imgIds=sorted(amodalGt.getImgIds())
    amodalDtFile = '/tmp/%s_amodalDt.json' %(''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6)))

    resFiles = []
    for filename in glob.glob(args.resFileFolder + '*.json'):
        resFiles.append(filename)
    if len(resFiles) == 0:
        print("wrong")
    assert len(resFiles) > 0, " wrong resFileFolder."
            
    amodalDt = filterDtFile(resFiles, imgIds)
    with open(amodalDtFile, 'wb') as output:
        json.dump(amodalDt, output)
    amodalDt=amodalGt.loadRes(amodalDtFile)
    # both things and stuff
    metric10 = singleEval(amodalDt, amodalGt, onlyThings=0, maxProp=args.maxProp)
    # things only
    metric11 = singleEval(amodalDt, amodalGt, onlyThings=1, maxProp=args.maxProp)
    # amodalGT, stuff only
    metric12 = singleEval(amodalDt, amodalGt, onlyThings=2, maxProp=args.maxProp)
    
    metric10.summarize(args.outputFile)
    metric11.summarize(args.outputFile)
    metric12.summarize(args.outputFile)
        
    metrics = {}
    metrics['both'] = metric10
    metrics['things'] = metric11
    metrics['stuff'] = metric12
    
    os.system("rm -f " + amodalDtFile)
    print("done! intermediate file cleaned!")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--resFileFolder', required=True)
    parser.add_argument('--dataType', default='val2014')
    parser.add_argument('--dataDir', default='./')
    parser.add_argument('--maxProp', default=1000, type=int)
    parser.add_argument('--outputFile', default='', type=str)
    args = parser.parse_args()

    metrics = main(args)
