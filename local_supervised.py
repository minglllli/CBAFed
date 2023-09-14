import numpy as np
import torch
import torch.optim
from options import args_parser
import copy
from FedAvg import FedAvg, model_dist
from validation import epochVal_metrics_test
from utils import losses
import logging
#from pytorch_metric_learning import losses
from networks.models import ModelFedCon
import torchvision.models as torch_models
import torch.nn as nn
from validation import epochVal_metrics_test
from cifar_load import get_dataloader, partition_data, partition_data_allnoniid

args = args_parser()


def test(epoch, checkpoint, data_test, label_test, n_classes):
    if args.model == 'Res18':
        net = torch_models.resnet18(pretrained=args.Pretrained)
        net.fc = nn.Linear(net.fc.weight.shape[1], n_classes)
    if len(args.gpu.split(',')) > 1:
        net = torch.nn.DataParallel(net, device_ids=[i for i in range(round(len(args.gpu) / 2))])
    model = net.cuda()
    model.load_state_dict(checkpoint)

    if args.dataset == 'SVHN' or args.dataset == 'cifar100' or args.dataset == 'cifar10':
        test_dl, test_ds = get_dataloader(args, data_test, label_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True)
    elif args.dataset == 'skin' or args.dataset == 'STL10' or args.dataset == 'fmnist':
        test_dl, test_ds = get_dataloader(args, data_test, label_test,
                                          args.dataset, args.datadir, args.batch_size,
                                          is_labeled=True, is_testing=True, pre_sz=args.pre_sz, input_sz=args.input_sz)

    AUROCs, Accus = epochVal_metrics_test(model, test_dl, args.model, n_classes=n_classes)
    AUROC_avg = np.array(AUROCs).mean()
    Accus_avg = np.array(Accus).mean()

    return AUROC_avg, Accus_avg


class SupervisedLocalUpdate(object):
    def __init__(self, args, idxs, n_classes):
        self.epoch = 0
        self.iter_num = 0
        # self.confuse_matrix = torch.zeros((5, 5)).cuda()
        self.base_lr = args.base_lr
        self.data_idx = idxs
        self.max_grad_norm = args.max_grad_norm
        if args.model == 'Res18':
            net = torch_models.resnet18(pretrained=args.Pretrained)
            net.fc = nn.Linear(net.fc.weight.shape[1], n_classes)
        if len(args.gpu.split(',')) > 1:
            net = torch.nn.DataParallel(net, device_ids=[i for i in range(round(len(args.gpu) / 2))])
        self.model = net.cuda()

    def train(self, args, net_w, op_dict, dataloader, n_classes, is_test=False, local_w=None, X_test=None, y_test=None, res=False, stage=2):
        self.model.load_state_dict(copy.deepcopy(net_w))
        self.model.cuda().train()
        #trans.cuda().train()
        if args.opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.base_lr,
                                              betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(),
                                             lr=args.base_lr, momentum=0.9,
                                             weight_decay=5e-4)
        elif args.opt == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.base_lr,
                                               weight_decay=0.02)
        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr

        loss_fn = torch.nn.CrossEntropyLoss()
        epoch_loss = []
        dic = []
        test_acc = []
        test_acc_avg = []
        model_list = []
        #logging.info('Begin supervised training')
        # stage1: supervised training
        if stage==1:
            if args.dataset=='cifar100':
                s_epoch = 1001
            else:
                s_epoch = 501
        else:
            s_epoch = 11
        for epoch in range(s_epoch):
            self.model.train()
            batch_loss = []
            accuracy = []
            accuracy1 = []
            label_all = []
            for i, (_, image_batch, label_batch) in enumerate(dataloader):

                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                label_batch = label_batch.long().squeeze()
                inputs = image_batch
                outputs = self.model(inputs)

                if len(label_batch.shape) == 0:
                    label_batch = label_batch.unsqueeze(dim=0)
                if len(outputs.shape) != 2:
                    outputs = outputs.unsqueeze(dim=0)
                label_all.append(label_batch)
                loss_classification = loss_fn(outputs, label_batch)
                loss = loss_classification
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.max_grad_norm)
                self.optimizer.step()


                """outputs_trans = trans(image_batch_trans)
                if len(outputs_trans.shape) != 2:
                    outputs_trans = outputs_trans.unsqueeze(dim=0)
                loss_classification_trans = loss_fn(outputs_trans, label_batch)
                loss = loss_classification_trans
                optimizer_trans.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trans.parameters(),
                                               max_norm=self.max_grad_norm)"""
                #optimizer_trans.step()


                batch_loss.append(loss.item())
                self.iter_num = self.iter_num + 1
                with torch.no_grad():
                    accuracy.append((torch.argmax(outputs, dim=1)==label_batch).float().mean())

            self.epoch = self.epoch + 1
            label_all = torch.cat(label_all, dim=0)
            class_n = torch.zeros(n_classes)
            for i in range(n_classes):
                class_n[i] = (label_all==i).float().sum()
            #print(class_n)
            epoch_loss.append(np.array(batch_loss).mean())
            # weight connection with previous epoch
            res = True
            if res:
                if (epoch==0):
                    print('res weight connection')
                    record_w = (copy.deepcopy(self.model.cpu().state_dict()))
                    #w_ttt = [copy.deepcopy(self.model.cpu().state_dict())]
                    self.model.cuda()
                if (epoch>0 and epoch%5==0):
                    #print('average weight test')
                    """w_ttt.append(copy.deepcopy(self.model.cpu().state_dict()))
                    n_l = [1 for x in range(len(w_ttt))]
                    w = FedAvg(None, w_ttt, n_l)
                    n_model = n_model + 1"""

                    print('res weight connection')
                    w_l = [record_w, copy.deepcopy(self.model.cpu().state_dict())]
                    if args.dataset == 'skin':
                        if stage==1:
                            n_l = [4., 1.]
                        else:
                            n_l = [4., 1.]
                    else:
                        n_l = [4., 1.]
                    print(n_l)

                    w = FedAvg(record_w, w_l, n_l)
                    record_w = copy.deepcopy(w)
                    """if type(X_test) != type(None):
                        #w = self.model.cpu().state_dict()
                        AUROC_avg, Accus_avg = test(epoch, w, X_test, y_test, n_classes)
                        print(AUROC_avg, Accus_avg)
                        test_acc.append(Accus_avg)
                        print(test_acc)"""
                    self.model.load_state_dict(w)
                    self.model.cuda()
            """if (s_epoch-epoch<=20 and epoch%5==0):
                model_list.append(self.model.cpu().state_dict())
                self.model.cuda()
            if epoch%5==0 and type(X_test) != type(None):
                # w = self.model.cpu().state_dict()
                AUROC_avg, Accus_avg = test(epoch, self.model.cpu().state_dict(), X_test, y_test, n_classes)
                print(AUROC_avg, Accus_avg)
                test_acc.append(Accus_avg)
                print(test_acc)
                self.model.cuda()"""
            if epoch>0 and epoch%1==0 and type(X_test) != type(None):
                w = self.model.cpu().state_dict()
                AUROC_avg, Accus_avg = test(epoch, w, X_test, y_test, n_classes)
                print(epoch, AUROC_avg, Accus_avg)
                test_acc_avg.append(Accus_avg)
                print(test_acc_avg)
                self.model.cuda()

            if s_epoch == epoch+1:
                print(f'epoch:{epoch}, accuracy:{sum(accuracy)/len(accuracy)}')
           # print(self.optimizer.state_dict()['lr'])

       # trans.cpu()
        self.model.cpu()
        return self.model.state_dict(), sum(epoch_loss) / len(epoch_loss), copy.deepcopy(
            self.optimizer.state_dict())
