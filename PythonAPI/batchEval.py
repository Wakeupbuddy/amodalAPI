import os,sys,glob,json
import pdb
from pycocotools.amodal import Amodal
from myPyAmodalEvalDemo import evalWrapper, filterDtFile
dd = pdb.set_trace
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
        #self.ap_severe = 0
        self.ar_none = 0
        self.ar_partial = 0
        self.ar_heavy = 0
        #self.ar_severe = 0
        self.order_ar_05 = {} # random, score, oracle
        self.order_ar = {}
        self.maxProp = maxProp

    def summarize(self):
        print("")
        print("== " + self.name)
        print("AR1: " + mystr(self.ar1))
        print("AR10: " + mystr(self.ar10))
        print("AR100: " + mystr(self.ar100))
        if self.maxProp == 1000:
            print("AR1000: " + mystr(self.ar1000))

        print("AR_none: " + mystr(self.ar_none))
        print("AR_partial: " + mystr(self.ar_partial))
        print("AR_heavy: " + mystr(self.ar_heavy))
        #print("AR_severe: " + mystr(self.ar_severe))
        
        order_ar05_str = ''
        for key, val in self.order_ar_05.iteritems():
            order_ar05_str = order_ar05_str + " " + key + ":" + mystr(val)+ " -"
        print("order AR 0.5: " + order_ar05_str)
        
        order_ar_str = ''
        for key, val in self.order_ar.iteritems():
            order_ar_str = order_ar_str + " " + key + ":" + mystr(val)+ " -"
        print("order AR: " + order_ar_str)        

def singleEval(useAmodalGT, onlyThings,amodalDt, amodalGt, maxProp):
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
    
    sortKey = 'oracle'
    occRange = 'all'
    useAmodalDT = 0
    stats = evalWrapper(amodalDt, amodalGt, useAmodalGT, useAmodalDT, onlyThings, sortKey, occRange, maxProp)
    metric.ap = stats[0]
    metric.ap_05 = stats[1]
    metric.ap_075 = stats[2]
    metric.ar1 = stats[3]
    metric.ar10 = stats[4]
    metric.ar100 = stats[5]
    metric.ar1000 = stats[6]
    metric.order_ar['oracle'] = stats[7]
    metric.order_ar_05['oracle'] = stats[8]

    sortKey = 'oracle'
    occRange = 'none'
    stats = evalWrapper(amodalDt, amodalGt,useAmodalGT, useAmodalDT, onlyThings, sortKey, occRange, maxProp)
    metric.ap_none = stats[0]
    if maxProp == 100:
        metric.ar_none = stats[5]
    else:
        metric.ar_none = stats[6]
    del stats

    sortKey = 'oracle'
    occRange = 'partial'
    stats = evalWrapper(amodalDt, amodalGt,useAmodalGT, useAmodalDT, onlyThings, sortKey, occRange, maxProp)
    metric.ap_partial = stats[0]
    if maxProp == 100:
        metric.ar_partial = stats[5]
    else:
        metric.ar_partial = stats[6]
    del stats

    sortKey = 'oracle'
    occRange = 'heavy'
    stats = evalWrapper(amodalDt, amodalGt, useAmodalGT, useAmodalDT, onlyThings, sortKey, occRange, maxProp)
    metric.ap_heavy = stats[0]
    if maxProp == 100:
        metric.ar_heavy = stats[5]
    else:
        metric.ar_heavy = stats[6]
    del stats
    
    return metric

if __name__ == "__main__":
    
    resFileFolder = sys.argv[1]
    modelName = sys.argv[2]

    session = sys.argv[3]
    #session = 'full'
     
    dataDir = '../'
    #dataType = 'val2014'
    dataType = sys.argv[4]
    #maxProp = 1000
    maxProp = int(sys.argv[5])
    annFile = '%s/annotations/COCO_amodal_%s.json'%(dataDir,dataType)
    amodalGt=Amodal(annFile)
    imgIds=sorted(amodalGt.getImgIds())
    amodalDtFile = '/tmp/%s_%s_amodalDt.json' %(modelName, dataType)

    resFiles = []
    for filename in glob.glob(resFileFolder + '*.json'):
        resFiles.append(filename)
    if len(resFiles) == 0:
        print("wrong")
    assert len(resFiles) > 0, " wrong resFileFolder?"
            
    amodalDt = filterDtFile(resFiles, imgIds)
    with open(amodalDtFile, 'wb') as output:
        json.dump(amodalDt, output)
    amodalDt=amodalGt.loadRes(amodalDtFile)

    if session == 'full':
        metric10 = singleEval(1, 0, amodalDt, amodalGt, maxProp) # amodalGT, both things and stuff
        metric11 = singleEval(1, 1, amodalDt, amodalGt, maxProp) # amodalGT, things only
        metric12 = singleEval(1, 2, amodalDt, amodalGt, maxProp) # amodalGT, stuff only
        
        metric10.summarize()
        metric11.summarize()
        metric12.summarize()
    else:
        raise NotImplementedError
        
    os.system("rm -f " + amodalDtFile)
    print("done! intermediate file cleaned!")
