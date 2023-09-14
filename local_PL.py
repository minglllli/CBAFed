from torch.utils.data import Dataset
import copy
import torch
import torch.optim
import torch.nn.functional as F
from options import args_parser
from networks.models import ModelFedCon
from utils import losses, ramps
from FedAvg import FedAvg, model_dist
import torch.nn as nn
import numpy as np
import  random
import logging
from torchvision import transforms
from ramp import LinearRampUp
import torchvision.models as torch_models
import torch.nn as nn
args = args_parser()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


# alpha=0.999
def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        items, index, weak_aug, strong_aug, label = self.dataset[self.idxs[item]]
        return items, index, weak_aug, strong_aug, label


class PLUpdate(object):
    def __init__(self, args, idxs, n_classes):
        if args.model == 'Res18':
            net = torch_models.resnet18(pretrained=args.Pretrained)
            net.fc = nn.Linear(net.fc.weight.shape[1], n_classes)
        self.model = net.cuda()
        self.data_idxs = idxs
        self.epoch = 0
        self.iter_num = 0
        self.flag = True
        self.pl_lr = args.pl_lr
        self.softmax = nn.Softmax()
        self.max_grad_norm = args.max_grad_norm
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.max_step = args.rounds * round(len(self.data_idxs) / args.batch_size)
        if args.opt == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.unsup_lr,
                                              betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.unsup_lr, momentum=0.9,
                                             weight_decay=5e-4)
        elif args.opt == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.unsup_lr,
                                               weight_decay=0.02)
        #self.max_warmup_step = round(len(self.data_idxs) / args.batch_size) * args.num_warmup_epochs
        #self.ramp_up = LinearRampUp(length=self.max_warmup_step)

    def train(self, args, net_w, op_dict, train_dl_local, n_classes, is_train=True, class_confident=None, include_second=False, second_c=None, second_h=None):
        self.model.load_state_dict(copy.deepcopy(net_w))
        temp_sss = copy.deepcopy(net_w)
        self.model.train()
        self.model.cuda()
        include_second = True
        #include_second = False
        """if args.dataset == 'SVHN' or args.dataset == 'skin':
            print('not include')
            include_second = False
        else:
            print('include second')
            include_second = True"""

        self.optimizer.load_state_dict(op_dict)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.pl_lr

        loss_fn = torch.nn.CrossEntropyLoss()
        epoch_loss = []
        #logging.info('Unlabeled client %d begin pl training' % unlabeled_idx)
        for i in range(1):
            iteration_accuracy = []

            train_data = []
            train_data_1 = []
            train_label = []
            new_data = []
            new_label = []
            # correct pseudo labels after threshold
            correct_pseu = 0
            # number of pseudo labels after threshold
            all_labels = []
            # correct pseudo labels without threshold
            train_right = 0
            num = 0
            total = 0
            true_label = []
            label_pl = []
            pseu_all = []
            #pl_us_p = []
            #pl_us_c = []
            #pl_wr_c = []
            #pl_wr_p = []

            unselected_samples = []
            second = []
            second_true = []
            predicted_accuracy = []
            for i, (_, weak_image_batch, label_batch) in enumerate(train_dl_local):
                # obtain pseudo labels
                with torch.no_grad():
                    image_batch = weak_image_batch[0]
                    #strong_image_batch = weak_image_batch[1]
                    total = total + len(image_batch)
                    all_labels.append(label_batch)
                    image_batch = image_batch.cuda()
                   # label_batch = label_batch.long().cuda()
                    #self.model.train()
                    self.model.eval()
                    outputs = self.model(image_batch)

                    if len(label_batch.shape) == 0:
                        label_batch = label_batch.unsqueeze(dim=0)
                    if len(outputs.shape) != 2:
                        outputs = outputs.unsqueeze(dim=0)

                    guessed = F.softmax(outputs, dim=1).cpu()
                    pseu = torch.argmax(guessed, dim=1).cpu()
                    label = label_batch.squeeze()
                    confident_threshold = torch.zeros(pseu.shape)
                    for i in range(len(pseu)):
                        confident_threshold[i] = class_confident[pseu[i]]
                    if len(label.shape) == 0:
                        label = label.unsqueeze(dim=0)

                    correct_pseu += torch.sum(label[torch.max(guessed, dim=1)[0] > confident_threshold] == pseu[
                        torch.max(guessed, dim=1)[0] > confident_threshold].cpu()).item()
                    train_right += sum([pseu[i].cpu() == label_batch[i].int() for i in range(label_batch.shape[0])])

                #label_us = label[torch.max(guessed, dim=1)[0] <= args.confidence_threshold]
                ##n = pseu[torch.max(guessed, dim=1)[0] <= args.confidence_threshold]
                #un_p = guessed[torch.max(guessed, dim=1)[0] <= args.confidence_threshold]
                #pl_us_c.append(un[label_us==un])
                #pl_us_p.append(un_p[label_us == un])

                pl = pseu[torch.max(guessed, dim=1)[0] > confident_threshold]
                pseu_all.append(pseu)
                num = num + len(pl)
                select_samples = image_batch[torch.max(guessed, dim=1)[0] > confident_threshold]
                uns_p = guessed[torch.max(guessed, dim=1)[0] <= confident_threshold]
                uns_samples = image_batch[torch.max(guessed, dim=1)[0] <= confident_threshold]
                uns_p_true = label_batch[torch.max(guessed, dim=1)[0] <= confident_threshold]
                if include_second:
                    pl_u = []
                    sample_u = []
                    for i in range(len(uns_p)):
                        p = uns_p[i]
                        index_max = p.argmax()
                        max_p = max(p)
                        p[p.argmax()] = 0
                        if p.argmax() in second_c:
                            second.append(p.argmax())
                            second_true.append(uns_p_true[i].cpu().numpy())
                            sample_u.append(uns_samples[i].cpu().numpy())
                            pl_u.append(p.argmax())
                    pl_u = torch.tensor(pl_u).long()
                    sample_u = torch.tensor(sample_u).cuda()
                    if(len(sample_u.shape)==3):
                        sample_u = sample_u.reshape(1, *sample_u.shape)

                    #train_data_1.append(strong_image_batch[torch.max(guessed, dim=1)[0] > args.confidence_threshold])
                unselected_samples.append(image_batch[torch.max(guessed, dim=1)[0] <= confident_threshold])
                #pseudo_p_s.append(guessed[torch.max(guessed, dim=1)[0] > args.confidence_threshold])
                #pseudo_p_u.append(guessed[torch.max(guessed, dim=1)[0] <= args.confidence_threshold])
                train_label.append(pl)
                train_data.append(select_samples)
                if include_second and len(pl_u) != 0:
                    train_data.append(sample_u)
                    train_label.append(pl_u)
                    #new_data.append(sample_u)
                    #new_label.append(pl_u)
                label_pl.append(label[torch.max(guessed, dim=1)[0] > confident_threshold])
                true_label.append(label)
                self.iter_num = self.iter_num + 1
           # print(epoch, sum(batch_loss)/len(batch_loss))
            predicted_accuracy.append(correct_pseu/num)
            print(f'selected number{num}, correctly predicted number{train_right.item()}, correct number{correct_pseu}, accuracy of selected number{correct_pseu/num}')





            train_data = torch.cat(train_data, dim=0)
            train_label = torch.cat(train_label, dim=0)
            """if (len(new_label)>1):
                new_data = torch.cat(new_data, dim=0)
                new_label = torch.cat(new_label, dim=0)
            elif(len(new_label)==1):
                new_data = new_data[0]
                new_label = new_label[0]"""
            pseu_all = torch.cat(pseu_all, dim=0)
            #train_data_1 = torch.cat(train_data_1, dim=0)
            label_pl = torch.cat(label_pl, dim=0)
            true_label = torch.cat(true_label, dim=0)
            class_num = torch.zeros(n_classes)
            class_num_r = torch.zeros(n_classes)
            class_num_w = torch.zeros(n_classes)
            class_true = torch.zeros(n_classes)
            all_pseu = torch.zeros(n_classes)
            for i in range(n_classes):
                class_num[i] = (train_label==i).float().sum()
                #class_num_r[i] = (train_label[train_label==label_pl]==i).float().sum()
                #class_num_w[i] = (train_label[train_label != label_pl] == i).float().sum()
                class_true[i] = (true_label==i).float().sum()
                all_pseu[i] = (pseu_all==i).float().sum()


            r_n = ((torch.tensor(second_true).reshape(len(second_true))==torch.tensor(second)).float().sum())
            #print(r_n, len(second_true), r_n/len(second_true))
            print(f'second labels:{len(second)}, accuracy:{r_n/len(second_true)}')
            #print(f'all pseudo labels: {all_pseu}')
            if is_train:
                for j in range(1):
                    for i in range(0, len(train_data), args.batch_size):
                        self.model.train()
                        data_batch = train_data[i:min(len(train_data), i+args.batch_size)].cuda()
                        if(len(data_batch)==1):
                            continue
                        #data_batch_1 = train_data_1[i:min(len(train_data), i+args.batch_size)].cuda()
                        label_batch = train_label[i:min(len(train_label), i + args.batch_size)].cuda()
                        #label_true = true_label[i:min(len(train_label), i + args.batch_size)].cpu()
                        outputs = self.model(data_batch)
                        #__1, activations_1, outputs_1 = self.model(data_batch, model=args.model)
                        if len(label_batch.shape) == 0:
                            label_batch = label_batch.unsqueeze(dim=0)
                        if len(outputs.shape) != 2:
                            outputs = outputs.unsqueeze(dim=0)


                        loss_classification = loss_fn(outputs, label_batch)
                        #loss = loss_classification + 0.5*loss_2
                        loss = loss_classification
                        self.optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       max_norm=self.max_grad_norm)
                        self.optimizer.step()
                #batch_loss.append(loss.item().mean())
       # print(sum(batch_loss) / len(batch_loss).item())

        #epoch_loss.append(sum(batch_loss) / len(batch_loss))
        self.epoch = self.epoch + 1
        self.model.cpu()
        return self.model.state_dict(), copy.deepcopy(self.optimizer.state_dict()), sum(class_num), class_num