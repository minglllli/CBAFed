# import os
# import sys
#
# import shutil
# import argparse
# import logging
# import time
# import random
# import numpy as np
# import pandas as pd

import torch
from torch.nn import functional as F
from utils.metrics import  compute_metrics_test

def epochVal_metrics_test(model, dataLoader,model_type,n_classes):
    training = model.training
    model.eval()

    gt = torch.FloatTensor().cuda()
    pred = torch.FloatTensor().cuda()
    
    gt_study   = {}
    pred_study = {}
    studies    = []

    with torch.no_grad():
        for i, (study, image, label) in enumerate(dataLoader):
            # study is the
            image, label = image.cuda(), label.cuda()
            output = model(image)
            # _:intermediate feature, feature: second last layer output, output: last layer output
            study = study.tolist()
            output = F.softmax(output, dim=1)

            for i in range(len(study)):
                if study[i] in pred_study:
                    assert torch.equal(gt_study[study[i]], label[i])
                    pred_study[study[i]] = torch.max(pred_study[study[i]], output[i])
                else:
                    gt_study[study[i]] = label[i]
                    pred_study[study[i]] = output[i]
                    studies.append(study[i])

          
        
        for study in studies:
            gt = torch.cat((gt, gt_study[study].view(1, -1)), 0)
            pred = torch.cat((pred, pred_study[study].view(1, -1)), 0)
        #gt=F.one_hot(gt.to(torch.int64).squeeze())
        #AUROCs, Accus, Senss, Specs, pre, F1 = compute_metrics_test(gt, pred,  thresh=thresh, competition=True)
        AUROCs, Accus, Pre, Recall = compute_metrics_test(gt, pred, n_classes=n_classes)

    model.train(training)

    return AUROCs, Accus
        #, Pre, Recall#,all_features.cpu(),all_labels.cpu()#, Senss, Specs, pre,F1
