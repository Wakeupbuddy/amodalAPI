__author__ = 'yzhu'

import numpy as np
import datetime
import time
from collections import defaultdict
import json
import mask
import copy
import pdb
import torchfile
import operator
import os.path
dd = pdb.set_trace
class AmodalEval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  useSegm    - [1] if true evaluate against ground-truth segments
    #  useCats    - [1] if true use category labels for evaluation    # Note: if useSegm=0 the evaluation is run on bounding boxes.
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, amodalGt=None, amodalDt=None):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param amodalGt: amodal object with ground truth annotations
        :param amodalDt: amodal object with detection results(no category)
        :return: None
        '''
        self.amodalGt   = amodalGt              # ground truth amodal API
        self.amodalDt   = amodalDt              # detections amodal API
        self.params   = {}                  # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params()              # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not amodalGt is None:
            self.params.imgIds = sorted(amodalGt.getImgIds())
            self.params.catIds = [1]

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        #
        def _toMask(objs, coco):
            # modify segmentation by reference
            for obj in objs:
                t = coco.imgs[obj['image_id']]
                for region in obj['regions']:
                    if type(region['segmentation']) == list:
                        # format is xyxy, convert to RLE
                        region['segmentation'] = mask.frPyObjects([region['segmentation']], t['height'], t['width'])
                        if len(region['segmentation']) == 1:
                            region['segmentation'] = region['segmentation'][0]
                        else:
                            # an object can have multiple polygon regions
                            # merge them into one RLE mask
                            dd()
                            region['segmentation'] = mask.merge(obj['segmentation'])
                        if 'area' not in region:
                            dd()
                            region['area'] = mask.area([region['segmentation']])[0]
                    elif type(region['segmentation']) == dict and type(region['segmentation']['counts']) == list:
                        dd()
                        region['segmentation'] = mask.frPyObjects([region['segmentation']],t['height'],t['width'])[0]
                    elif type(region['segmentation']) == dict and \
                        type(region['segmentation']['counts'] == unicode or type(region['segmentation']['counts']) == str):
                        # format is already RLE, do nothing
                        if 'area' not in region:
                            region['area'] = mask.area([region['segmentation']])[0]
                    else:
                        raise Exception('segmentation format not supported.')
        p = self.params
        gts=self.amodalGt.loadAnns(self.amodalGt.getAnnIds(imgIds=p.imgIds))
        dts=self.amodalDt.loadAnns(self.amodalDt.getAnnIds(imgIds=p.imgIds))

        if p.useSegm:
            _toMask(dts, self.amodalDt)
            _toMask(gts, self.amodalGt)
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
#            self._gts[gt['image_id'], gt['category_id']].append(gt)
            self._gts[gt['image_id'], 1].append(gt)
        for dt in dts:
            if 'category_id' not in dt:
                dt['category_id'] = 1
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = []   # per-image per-category evaluation results
        self.queries = []
        self.eval     = {}   # accumulated evaluation results
    
    def _findRegion(self, ann, uid):
        ret = []
        for region in ann['regions']:
            if region['id'] == uid:
                ret.append(region)
        assert(len(ret) < 2)
        if len(ret) == 1:
            return ret[0]
        else:
            return None
    # DEPRECIATED
    def _checkOrder(self, dt_front, dt_back, criterion='score'):
        # given two dts, where the ground truth is dt_front in front of dt_back, return yes if 
        # the criterion agrees with the ground truth
        if criterion == 'score':
            return dt_front[criterion] > dt_back[criterion]
        if criterion == 'random':
            return np.random.rand() > 0.5
        if criterion == 'oracle': # to check the upperbound
            return True
    # DEPRECIATED
    def _checkOrderV2(self, dt_front, dt_back, amodalIntersect, criterion='invisiblePixels'):
        # given two dts, where the ground truth is dt_front in front of dt_back, return yes if 
        # the criterion agrees with the ground truth
        if criterion == 'invisiblePixels':
            inv_front = mask.merge([dt_front['invisible_mask'], amodalIntersect],1)
            inv_back = mask.merge([dt_back['invisible_mask'], amodalIntersect],1)
            inv_area_front = mask.area([inv_front])[0]    
            inv_area_back = mask.area([inv_back])[0]
            
            #return dt_front['score'] > dt_back['score']
            if inv_area_front == 0 and inv_area_back == 0:
                return dt_front['score'] > dt_back['score']
            else:
                return inv_area_front < inv_area_back

        elif criterion == 'AmodalOracle':
            return True
        elif criterion == 'stillScore':
            return dt_front['score'] > dt_back['score']
        elif criterion == 'area':
            a1 = mask.area([dt_front['segmentation']])[0]
            a2 = mask.area([dt_back['segmentation']])[0]
            return a1 < a2
        elif criterion == 'yaxis':
            a1 = mask.decode([dt_front['segmentation']])
            ysum1 = a1.sum(1)
            lowest_y1 = self._findLowest(ysum1)
            highest_y1 = self._findHighest(ysum1)
            a2 = mask.decode([dt_back['segmentation']])
            ysum2 = a2.sum(1)
            lowest_y2 = self._findLowest(ysum2)
            highest_y2 = self._findHighest(ysum2)
            return highest_y1 > highest_y2 # the front object should have lower highest y
            #return lowest_y1 > lowest_y2 # the front object should have lower lowest y.

    def _findLowest(self, ysum):
        # given a vector, return the largst nonzero index
        for i in range(ysum.shape[0]):
            if ysum[ysum.shape[0]-i-1] > 0:
                return ysum.shape[0] - i - 1
        print('oops, bug in ysum')
        dd()

    def _findHighest(self, ysum):
        # given a vector, return the smallest nonzero index
        for i in range(ysum.shape[0]):
            if ysum[i] > 0:
                return i
        print('oops, bug in ysum')
        dd()

    def _filterConstraint_super(self, g_front, g_back, gt):
        if self.params.onlyThings != 3:
            return self._filterConstraint(g_front, gt) == False or self._filterConstraint(g_back, gt)==False
        else:
            isStuff1 = gt['regions'][g_front-1]['isStuff']
            isStuff2 = gt['regions'][g_back-1]['isStuff']
            if isStuff1 + isStuff2 == 1:
                return True
            else:
                return False
            
    def _filterConstraint(self, g_id, gt):
        isStuff = gt['regions'][g_id-1]['isStuff']
        if self.params.onlyThings == 2 and isStuff == 0:
            return False
        elif self.params.onlyThings == 1 and isStuff == 1:
            return False
        return True


    def evaluateDepthOrdering(self, evalRes):
        # load all the depth ordering constraint, then for each constraint, check the corresponding dts,
        # then for each pair of dt, use lambda function to determine who's in front.
        correctOrder = [0] * len(self.params.iouThrs)
        self.qs = {}
        for i, thr in enumerate(self.params.iouThrs):
            self.qs[i] = []
        gt = self._gts[evalRes['image_id'], 1][0] # for multi gt, simply use first annotation. fixme
        if ',' not in gt['depth_constraint']:
            #print("no constraint here")
            return correctOrder, 0
        gtConstraints = gt['depth_constraint'].split(',')
        constraint_cnt = 0
        for icon, constrain in enumerate(gtConstraints):
            try:
                g_front, g_back = constrain.split('-')
            except:
                print("error in split")
                dd()
            g_front = int(g_front); g_back = int(g_back)
            if self._filterConstraint_super(g_front, g_back, gt):
            #if self._filterConstraint(g_front, gt) == False or self._filterConstraint(g_back, gt)==False:
                # possible to filter out stuff, and only consider between things constraints
                continue
            
            constraint_cnt = constraint_cnt + 1
            for i, thr in enumerate(self.params.iouThrs):
                # at IoU threshold thr, find matched gt for g_front and g_back in gtMatches
                # Then use these 2 dt id, find their mask and check overlap
                
                # notice this result always use the max value in maxDet (default) (fixme?)
                dt_front_id = int(evalRes['gtMatches'][:,g_front-1][i]) # float to int. fixme
                dt_back_id = int(evalRes['gtMatches'][:,g_back-1][i])
                if dt_front_id == 0 or dt_back_id == 0:
                    continue

                dt_front = self._findRegion(self._dts[evalRes['image_id'], 1][0], dt_front_id)
                dt_back = self._findRegion(self._dts[evalRes['image_id'], 1][0], dt_back_id)
                if dt_front == None or dt_back == None:
                    print("wrong region id")
                    dd()

                if 'amodal_mask' in dt_front:
                    dt_front_shape = dt_front['amodal_mask']
                else:
                    dt_front_shape = dt_front['segmentation']

                if 'amodal_mask' in dt_back:
                    dt_back_shape = dt_back['amodal_mask']
                else:
                    dt_back_shape = dt_back['segmentation']
                
                amodalIntersect = mask.merge([dt_front_shape, dt_back_shape],1)
                area = mask.area([amodalIntersect])[0]
                if self.params.sortKey.startswith('queryNN'):
                    if area > 0:
                        query = {}
                        query['dt_front'] = dt_front_shape
                        query['dt_back'] = dt_back_shape
                        query['image_id'] = evalRes['image_id']
                        self.qs[i].append(query)
                if self.params.sortKey == 'invisiblePixels' or self.params.sortKey == 'AmodalOracle' or self.params.sortKey == 'stillScore' or self.params.sortKey == 'area' or self.params.sortKey == 'yaxis':
                    #amodalIntersect = mask.merge([dt_front_shape, dt_back_shape],1)
                    #area = mask.area([amodalIntersect])[0]
                    if area > 0 and self._checkOrderV2(dt_front, dt_back, amodalIntersect, self.params.sortKey):
                        correctOrder[i] = correctOrder[i] + 1
                else:
                    #area = mask.area([mask.merge([dt_front_shape, dt_back_shape],1)])[0]
                    if area > 0 and self._checkOrder(dt_front, dt_back, self.params.sortKey):
                        correctOrder[i] = correctOrder[i] + 1

        return correctOrder, constraint_cnt
       

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        #print 'Running per image evaluation...'
        p = self.params
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p
        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]
        computeIoU = self.computeIoU
        evaluateDepthOrdering = self.evaluateDepthOrdering
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}
        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        for catId in catIds:
            for areaRng in p.areaRng:
                for imgId in p.imgIds:
                    evalRes = evaluateImg(imgId, catId, areaRng, maxDet, p.occRng)
                    evalRes['correctOrder'], evalRes['totalOrder'] = self.evaluateDepthOrdering(evalRes)
                    self.evalImgs.append(evalRes)
                    if p.sortKey.startswith('queryNN'):
                        self.queries.append(self.qs)
        # DEPRECIATED 
        if p.sortKey.startswith('queryNN'):
            os.system('rm -f /tmp/queries.json')
            with open('/tmp/queries.json', 'w') as outfile:
                json.dump(self.queries, outfile)
            featureType = p.sortKey.split('-')
            inferCmd = 'th /media/denny1108/EA165C93165C631D/yzhu/deepmask/orderWrapper.lua'
            if type(featureType) == list and len(featureType)==2:
                inferCmd = inferCmd + ' -FeatureType ' + featureType[-1]
            os.system(inferCmd)
            if os.path.isfile('/tmp/answer.t7'):
                answer = torchfile.load('/tmp/answer.t7')
                os.system('rm -f /tmp/answer.t7')
            else:
                print("t7 file nonexist!")
                dd()
            cnt = 0
            imgCnt = 0
            for catId in catIds: #assume flat, fixme
                for areaRng in p.areaRng: #assume flat, fixme
                    for imgId in p.imgIds: # 1323
                        #print(imgCnt)
                        for ithr, thr in enumerate(self.params.iouThrs):
                            self.evalImgs[imgCnt]['correctOrder'][ithr] = answer[cnt]['correctOrder']
                            cnt = cnt + 1
                        #print(self.evalImgs[imgCnt]['correctOrder'])
                        #print(self.evalImgs[imgCnt]['totalOrder'])
                        imgCnt = imgCnt + 1
            #print(cnt)
       
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print 'DONE (t=%0.2fs).'%(toc-tic)

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        #assert(len(dt) == 1)
        # for multiple gt/annotator case, clowntownly use gt[0] for now. Fixme.
        dt = dt[0]['regions']; gt = gt[0]['regions']
        dt = sorted(dt, key=lambda x: -x['score'])
        if len(dt) > p.maxDets[-1]:
            dt=dt[0:p.maxDets[-1]]

        if p.useSegm:
            if p.useAmodalGT:
                g = [g['segmentation'] for g in gt]
            else:
                g = [g['visible_mask'] if 'visible_mask' in g else g['segmentation'] for g in gt]
            
            if p.useAmodalDT:
                d = [d['amodal_mask'] if 'amodal_mask' in d else d['segmentation'] for d in dt]
            else:
                d = [d['segmentation'] for d in dt]
        else:
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        # compute iou between each dt and gt region
        iscrowd = [0 for o in gt]
        ious = mask.iou(d,g,iscrowd)
        return ious

    def exportDtFile(self, fname):
        # save the matched dt, as a field of gt's regions. Then export the file again. 
        # This file will be used to visualize detection results, will be good for debugging
        if not self.evalImgs:
            print 'Please run evaluate() first'
        res = []
        for key, item in self._gts.iteritems():
            gt = item
            while type(gt) == list:
                gt = gt[0]
            
            if type(gt) != dict:
                dd()
            res.append(gt)
        with open(fname, 'wb') as output:
            json.dump(res, output)
        return res

    def evaluateImg(self, imgId, catId, aRng, maxDet, oRng):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        #
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None
        #assert(len(dt) == 1)
        dt = dt[0]['regions']; gt = gt[0]['regions']
        for g in gt:
            if 'ignore' not in g:
                g['ignore'] = 0
            g['_ignore'] = 0 # default
            if p.onlyThings == 1 and g['isStuff'] == 1:
                g['_ignore'] = 1
            if p.onlyThings == 2 and g['isStuff'] == 0:
                g['_ignore'] = 1
            if g['occlude_rate'] < oRng[0] or g['occlude_rate'] > oRng[1]:
                g['_ignore'] = 1

        # sort dt highest score first, sort gt ignore last
        # gt = sorted(gt, key=lambda x: x['_ignore'])
        gtind = [ind for (ind, g) in sorted(enumerate(gt), key=lambda (ind, g): g['_ignore']) ]
        def inv(perm):
            inverse = [0] * len(perm)
            for i, p in enumerate(perm):
                inverse[p] = i
            return inverse
        inv_gtind = inv(gtind)
        
        gt = [gt[ind] for ind in gtind]
        dt = sorted(dt, key=lambda x: -x['score'])[0:maxDet]
        iscrowd = [0 for o in gt]
        # load computed ious
        N_iou = len(self.ious[imgId, catId])
        ious = self.ious[imgId, catId][0:maxDet, np.array(gtind)] if N_iou >0 else self.ious[imgId, catId]
        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['order']
                    #dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]  = d['id']
        
        gtm = gtm[:,np.array(inv_gtind)]
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        # save matching ids into self._gts
        self._gts[imgId, catId][0]['gtm'] = gtm.tolist()
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['order'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        #print 'Accumulating evaluation results...   '
        tic = time.time()
        if not self.evalImgs:
            print 'Please run evaluate() first'
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        orderRecall = np.zeros((T,K,A,M))
        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        # K0 = len(_pe.catIds)
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list): # length(1)
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list): # length(1)
                Na = a0*I0
                for m, maxDet in enumerate(m_list): # length(4)
                    E = [self.evalImgs[Nk+Na+i] for i in i_list]
                    E = filter(None, E)
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')

                    #dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)
                    #dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)
                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore']  for e in E])
                    npig = len([ig for ig in gtIg if ig == 0])
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )
                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

                    totalOrder = 0
                    correctOrder = [0] * len(p.iouThrs)
                    for e in E:
                        totalOrder = totalOrder + e['totalOrder']
                        for t, thr in enumerate(p.iouThrs):
                            correctOrder[t] = correctOrder[t] + e['correctOrder'][t]
                    
                    for t, thr in enumerate(p.iouThrs):
                        orderRecall[t, k, a, m] = (correctOrder[t] + 0.0)/totalOrder

                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        
                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs)
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)

        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'precision': precision,
            'recall':   recall,
            'orderRecall': orderRecall,
        }
        toc = time.time()
        #print 'DONE (t=%0.2fs).'%( toc-tic )

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr        = ' {:<18} {} @[ IoU={:<9} | area={:>6} | maxDets={:>3} ] = {}'
            #titleStr    = 'Average Precision' if ap == 1 else 'Average Recall'
            if ap == 1:
                typeStr = '(AP)'
                titleStr = 'Average Precision'
            elif ap == 2:
                typeStr = '(AR)'
                titleStr = 'Average Recall'
            else:
                typeStr = '(Order AR -- ' + p.sortKey + ')'
                titleStr = 'Order Average Recall'
            
            iouStr      = '%0.2f:%0.2f'%(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else '%0.2f'%(iouThr)
            areaStr     = areaRng
            maxDetsStr  = '%d'%(maxDets)

            aind = [i for i, aRng in enumerate(['all', 'small', 'medium', 'large']) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                # areaRng
                s = s[:,:,:,aind,mind]
            elif ap == 2:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                s = s[:,:,aind,mind]
            elif ap == 3:
                # orderRecall
                s = self.eval['orderRecall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print iStr.format(titleStr, typeStr, iouStr, areaStr, maxDetsStr, '%.3f'%(float(mean_s)))
            return mean_s

        if not self.eval:
            raise Exception('Please run accumulate() first')
        maxProp = self.params.maxDets[-1]  
        
        self.stats = np.zeros((12,))
        self.stats[0] = _summarize(1, maxDets = maxProp)
        self.stats[1] = _summarize(1,iouThr=.5, maxDets = maxProp)
        self.stats[2] = _summarize(1,iouThr=.75, maxDets = maxProp)
        self.stats[3] = _summarize(2,maxDets=1)
        self.stats[4] = _summarize(2,maxDets=10)
        self.stats[5] = _summarize(2,maxDets=100)
        if maxProp == 1000:
            self.stats[6] = _summarize(2,maxDets=1000)
        
        # depth ordering eval
        
        self.stats[7] = _summarize(3, maxDets = maxProp)
        self.stats[8] = _summarize(3, iouThr=.5, maxDets = maxProp)

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def __init__(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95-.5)/.05)+1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00-.0)/.01)+1, endpoint=True)
        self.maxDets = [1,10,100]
        #self.maxDets = [1,10,100,1000]
#        self.areaRng = [ [0**2,1e5**2], [0**2, 32**2], [32**2, 96**2], [96**2, 1e5**2] ]
        self.areaRng = [ [0**2,1e5**2] ]
        self.useSegm = 1
        self.useAmodalGT = 1
        self.onlyThings = 1 # 1: things only; 0: both
        self.useCats = 1
        self.occRng = [0, 1] # occluding ratio filter. not yet support multi filter for now.
        self.sortKey = 'score' # the criterion to determine dt's depth ordering
