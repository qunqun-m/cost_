import math
import numpy as np
import torch

def presision(result_list, gt_list, top_k):
    result = []
    for topk in top_k:
        count=0.0
        for r,g in zip(result_list,gt_list):
            r_ = r[:topk]
            count+=len(set(r_).intersection(set(g)))
        result.append(count/(topk*len(result_list)))
    return result

def recall(result_list, gt_list, top_k):
    result = []
    for topk in top_k:
        t=0.0
        for r,g in zip(result_list,gt_list):
            r_ = r[:topk]
            t+=1.0*len(set(r_).intersection(set(g)))/len(g)
        result.append(t/len(result_list))
    return result

def f_measure(result_list,gt_list,top_k,eps=1.0e-9):
    result = []
    for topk in top_k:
        f=0.0
        for r,g in zip(result_list,gt_list):
            r_ = r[:topk]
            recc=1.0*len(set(r_).intersection(set(g)))/len(g)
            pres=1.0*len(set(r_).intersection(set(g)))/topk
            if recc+pres<eps:
                continue
            f+=(2*recc*pres)/(recc+pres)
        result.append(f/len(result_list))
    return result

def novelty(result_list,s_u,top_k):
    count=0.0
    for r,g in zip(result_list,s_u):
        count+=len(set(r)-set(g))
    return count/(top_k*len(result_list))

def hit_ratio(result_list,gt_list, top_k):
    result = []
    for topk in top_k:
        result_list_ = []
        for re in result_list:
            result_list_.append(re[:topk])
        intersetct_set=[len(set(r)&set(g)) for r,g in zip(result_list_,gt_list)]
        x = 1.0*sum(intersetct_set)/sum([len(gts) for gts in gt_list])
        result.append(x)
    return result

def NDCG(result_list,gt_list, top_k):
    result = []
    for topk in top_k:
        t=0.0
        for re,gt in zip(result_list,gt_list): # predict, label
            setgt=set(gt)
            re_ = re[:topk]
            indicator=np.asfarray([1 if r in setgt else 0 for r in re_])
            # print(indicator)
            sorted_indicator=indicator[indicator.argsort(-1)[::-1]]
            # print(sorted_indicator)
            if 1 in indicator:
                curr = np.sum(indicator / np.log2(1.0*np.arange(2,len(indicator)+ 2)))/\
                np.sum(sorted_indicator/np.log2(1.0*np.arange(2,len(indicator)+ 2)))
                t = t + curr
                # print(curr)
        x = t/len(gt_list)
        result.append(x)
    return result

def NDCG_recforest(result_list,gt_list):
    t=0.0
    for re,gt in zip(result_list,gt_list):
        setgt=set(gt)
        indicator=np.asfarray([1 if r in setgt else 0 for r in re])
        sorted_indicator = np.ones(min(len(setgt), len(re)))
        if 1 in indicator:
            t+=np.sum(indicator / np.log2(1.0*np.arange(2,len(indicator)+ 2)))/\
               np.sum(sorted_indicator/np.log2(1.0*np.arange(2,len(sorted_indicator)+ 2)))
    return t/len(gt_list)

def NDCG_comicrec(result_list,gt_list):
    t=0.0
    for re,gt in zip(result_list,gt_list):
        recall = 0
        dcg = 0.0
        setgt=set(gt)
        for no, iid in enumerate(re):
            if iid in setgt:
                recall += 1
                dcg += 1.0 / math.log(no + 2, 2)
        idcg = 0.0
        for no in range(recall):
            idcg += 1.0 / math.log(no + 2, 2)
        if recall > 0:
            t += dcg / idcg
    return t/len(gt_list)

def MAP(result_list,gt_list,topk):
    t=0.0
    for re,gt in zip(result_list,gt_list):
        setgt=set(gt)
        indicator=np.asfarray([1 if r in setgt else 0 for r in re])
        t+=np.mean([indicator[:i].sum(-1)/i for i in range(1,topk+1)],axis=-1)
    return t/len(gt_list)





