from validation import epochVal_metrics_test
from options import args_parser
import os
import sys
import logging
import random
import numpy as np
import copy
import datetime
from FedAvg import FedAvg, model_dist
from torch.nn import Linear
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
from networks.models import ModelFedCon
from dataloaders import dataset
from local_supervised import SupervisedLocalUpdate
from local_PL import PLUpdate
from tqdm import trange
from cifar_load import get_dataloader, partition_data, partition_data_allnoniid
import torchvision.models as torch_models
import torch.nn as nn
import timm
from torchvision.datasets import MNIST, STL10, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils

from torch.utils.tensorboard import SummaryWriter

#os.environ['CUDA_VISIBLE_DEVICES'] = '7'


from timm.models.vision_transformer import vit_tiny_patch16_224


def split(dataset, num_users):
    ## randomly split dataset into equally num_users parties
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def select_samlple(x,y,n_classes):
    class_n = torch.zeros(n_classes)
    for i in range(n_classes):
        class_n[i] = (y==i).float().sum()
    X_new = []
    Y_new = []
    select_n = torch.ones(n_classes)*min(class_n)
    select_list = random.shuffle([i for i in range(len(y))])
    print(select_list)
    #print(y.shape, select_list[i], y[select_list[i]])
    for i in range(len(x)):
        if select_n[y[select_list[i]]] > 0:
            X_new.append(x[select_list[i]])
            Y_new.append(y[select_list[i]])
            select_n[y[select_list[i]]] = select_n[y[select_list[i]]] - 1

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

if __name__ == '__main__':
    weight_path = '/home/ubuntu/federated_semi_supervised_learning/RSCFed-main/'
    args = args_parser()
    supervised_user_id = [0]
    unsupervised_user_id = list(range(len(supervised_user_id), args.unsup_num + len(supervised_user_id)))
    sup_num = len(supervised_user_id)
    unsup_num = len(unsupervised_user_id)
    total_num = sup_num + unsup_num

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    print(args.gpu)
    torch.cuda.set_device(0)
    time_current = 'attempt0'
    if args.log_file_name is None:
        args.log_file_name = 'log-%s' % (datetime.datetime.now().strftime("%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(filename=os.path.join(args.logdir, log_path), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(str(args))
    logger.info(time_current)
    if args.deterministic:
        print('deterministic operation')
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    if not os.path.isdir('tensorboard'):
        os.mkdir('tensorboard')

    snapshot_path = 'model/'
    if not os.path.isdir(snapshot_path):
        os.mkdir(snapshot_path)
    if args.dataset == 'SVHN':
        snapshot_path = 'model/SVHN/'
    if args.dataset == 'cifar100':
        snapshot_path = 'model/cifar100/'
    if args.dataset == 'skin':
        snapshot_path = 'model/skin/'
    if not os.path.isdir(snapshot_path):
        os.mkdir(snapshot_path)

    print('==> Reloading data partitioning strategy..')
    assert os.path.isdir('partition_strategy'), 'Error: no partition_strategy directory found!'

    partition = torch.load('partition_strategy/SVHN_noniid_10%labeled.pth')
    net_dataidx_map = partition['data_partition']
    X_train, y_train, X_test, y_test = partition_data_allnoniid(
        args.dataset, args.datadir, partition=args.partition, n_parties=total_num, beta=args.beta)

    X_train = X_train.transpose([0, 2, 3, 1])
    X_test = X_test.transpose([0, 2, 3, 1])
    n_classes = 10

    print(args.model)
    if args.model == 'Res18':
        net_glob = torch_models.resnet18(pretrained=args.Pretrained)
        net_glob.fc = nn.Linear(net_glob.fc.weight.shape[1], n_classes)
    if args.resume:
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('warmup/SVHN.pth')

        net_glob.load_state_dict(checkpoint['state_dict'])
    else:
        start_epoch = 0

    if len(args.gpu.split(',')) > 1:
        net_glob = torch.nn.DataParallel(net_glob, device_ids=[i for i in range(round(len(args.gpu) / 2))])  #

    net_glob.train()
    w_glob = net_glob.state_dict()

    w_locals = []
    w_locals_trans = []
    w_ema_unsup = []
    lab_trainer_locals = []
    unlab_trainer_locals = []
    pl_trainer_locals = []
    sup_net_locals = []
    sup_net_locals_trans = []
    unsup_net_locals = []
    unsup_net_locals_trans = []
    pl_net_locals = []
    pl_net_locals_trans = []
    sup_optim_locals = []
    sup_optim_locals_trans = []
    unsup_optim_locals = []
    pl_optim_locals = []
    pl_optim_locals_trans = []
    dist_scale_f = args.dist_scale
    total_lenth = sum([len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))])
    each_lenth = [len(net_dataidx_map[i]) for i in range(len(net_dataidx_map))]
    client_freq = [len(net_dataidx_map[i]) / total_lenth for i in range(len(net_dataidx_map))]
    #load supervised trainer
    for i in supervised_user_id:
        lab_trainer_locals.append(SupervisedLocalUpdate(args, net_dataidx_map[i], n_classes))
        w_locals.append(copy.deepcopy(w_glob))
        w_locals_trans.append(copy.deepcopy(w_locals_trans))
        sup_net_locals.append(copy.deepcopy(net_glob))
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(sup_net_locals[i].parameters(), lr=args.base_lr,
                                         betas=(0.9, 0.999), weight_decay=5e-4)
            optimizer_trans = torch.optim.Adam(sup_net_locals_trans[i].parameters(), lr=0.03,
                                         betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(sup_net_locals[i].parameters(), lr=args.base_lr, momentum=0.9,
                                        weight_decay=5e-4)
            optimizer_trans = torch.optim.SGD(sup_net_locals_trans[i].parameters(), lr=0.03, momentum=0.9,
                                        weight_decay=5e-4)
        elif args.opt == 'adamw':
            optimizer = torch.optim.AdamW(sup_net_locals[i].parameters(), lr=args.base_lr, weight_decay=0.02)
            optimizer_trans = torch.optim.AdamW(sup_net_locals_trans[i].parameters(), lr=args.base_lr, weight_decay=0.02)
        if args.resume:
            optimizer.load_state_dict(checkpoint['sup_optimizers'][i])
        sup_optim_locals.append(copy.deepcopy(optimizer.state_dict()))
        sup_optim_locals_trans.append(copy.deepcopy(optimizer_trans.state_dict()))

    # load pseudo labelling trainer
    for i in unsupervised_user_id:
        pl_trainer_locals.append(PLUpdate(args, net_dataidx_map[i], n_classes))
        w_locals.append(copy.deepcopy(w_glob))
        pl_net_locals.append(copy.deepcopy(net_glob))
        if args.opt == 'adam':
            optimizer = torch.optim.Adam(pl_net_locals[i - sup_num].parameters(), lr=args.unsup_lr,
                                         betas=(0.9, 0.999), weight_decay=5e-4)
            optimizer_trans = torch.optim.Adam(pl_net_locals_trans[i - sup_num].parameters(), lr=0.03,
                                         betas=(0.9, 0.999), weight_decay=5e-4)
        elif args.opt == 'sgd':
            optimizer = torch.optim.SGD(pl_net_locals[i - sup_num].parameters(),
                                        lr=args.unsup_lr, momentum=0.9,
                                        weight_decay=5e-4)
        elif args.opt == 'adamw':
            optimizer = torch.optim.AdamW(pl_net_locals[i - sup_num].parameters(), lr=args.unsup_lr,
                                          weight_decay=0.02)
            optimizer_trans = torch.optim.AdamW(pl_net_locals_trans[i - sup_num].parameters(), lr=0.03,
                                          weight_decay=0.02)
        pl_optim_locals.append(copy.deepcopy(optimizer.state_dict()))

    sup_p = torch.zeros(n_classes)
    record_accuracy = []

    # supervised training in labeled clients, change com_round if number of labeled clients > 1
    for com_round in trange(1):
        print("************* Communication round %d begins *************" % com_round)
        #print('upper bound')
        #print(f'upper bound of CNNs')
        w_l = []
        n_l = []

        for client_idx in supervised_user_id:
            loss_locals = []
            clt_this_comm_round = []
            w_per_meta = []
            local = lab_trainer_locals[client_idx]
            optimizer = sup_optim_locals[client_idx]
            #optimizer_trans = sup_optim_locals_trans[client_idx]

            #X_new, y_new = select_samlple(X_train[net_dataidx_map[client_idx]],y_train[net_dataidx_map[client_idx]],10)
            train_dl_local, train_ds_local = get_dataloader(args, X_train[net_dataidx_map[client_idx]],
                                                            y_train[net_dataidx_map[client_idx]],
                                                            args.dataset, args.datadir, args.batch_size,
                                                            is_labeled=True,
                                                            data_idxs=net_dataidx_map[client_idx],
                                                            pre_sz=args.pre_sz, input_sz=args.input_sz)
            w, loss, op = local.train(args, sup_net_locals[client_idx].state_dict(),
                                               optimizer,
                                      train_dl_local, n_classes, X_test=X_test, y_test=y_test, res=True,stage=1)  # network, loss, optimizer
            w_l.append(w)
            n_l.append(len(net_dataidx_map[client_idx]))
            sup_optim_locals[client_idx] = copy.deepcopy(op)
        w = FedAvg(net_glob.state_dict(), w_l, n_l)

        net_glob.load_state_dict(w)
        if com_round%10==0:
            AUROC_avg, Accus_avg = test(com_round, net_glob.state_dict(), X_test, y_test, n_classes)
            print(AUROC_avg, Accus_avg)
            record_accuracy.append(Accus_avg)
            # print('adding lambda')
            print(record_accuracy)
        for i in supervised_user_id:
            sup_net_locals[i].load_state_dict(w)

    net_glob.load_state_dict(w)
    torch.save({'state_dict': net_glob.state_dict()}, 'SVHN_res_500_Res18_beta0.8.pth')
    AUROC_avg, Accus_avg = test(com_round, net_glob.state_dict(), X_test, y_test, n_classes)
    print(AUROC_avg, Accus_avg)
    # load supervised pretrained models
    state = torch.load('SVHN_res_500_Res18_beta0.8.pth')
    w = state['state_dict']
    record_w = copy.deepcopy(w)
    net_glob.load_state_dict(w)
    for i in supervised_user_id:
        sup_net_locals[i].load_state_dict(w)
    for i in unsupervised_user_id:
        pl_net_locals[i - sup_num].load_state_dict(w)

    # pseudo labelling and training
    T_base = 0.84

    T_lower = 0.03

    T_higher = 0.1
    T_upper = 0.95
    all_local = []
    # load number of classes in labeled clients
    sup_label = torch.load('partition_strategy/svhn_beta0.8_sup.pth')
    temp_sup_label = copy.deepcopy(sup_label)
    temp_sup_label = (temp_sup_label / sum(temp_sup_label))*(n_classes / 10)
    second_class = []
    second_h = []
    for i in range(len(temp_sup_label)):
        if temp_sup_label[i]<T_lower:
            second_class.append(i)
        if temp_sup_label[i]>T_higher:
            second_h.append(i)
    if min(temp_sup_label) < T_lower:
        include = True
    else:
        include = False
    temp_sup_label = temp_sup_label
    class_confident = temp_sup_label + T_base - temp_sup_label.std()
    if args.dataset == 'skin' or args.dataset == 'SVHN':
        class_confident[class_confident >= 0.9] = 0.9
    else:
        class_confident[class_confident >= T_upper] = T_upper
    print(class_confident)
    record_accuracy = []
    predict_accuracy = []
    sc = 10
    if args.dataset == 'cifar100':
        total_epoch = 1001
    else:
        total_epoch = 501

    for com_round in trange(total_epoch):
        temp_p_a = []
        print("************* Communication round %d begins *************" % com_round)
        print(f"Threshold base is {T_base}")
        print(f'second base is {T_lower}')
        #print(f'upper bound 0.99')
        local_w = []
        local_num = []
        local_label = torch.zeros(n_classes)

        for client_idx in supervised_user_id:
            loss_locals = []
            clt_this_comm_round = []
            w_per_meta = []
            local = lab_trainer_locals[client_idx]
            optimizer = sup_optim_locals[client_idx]
            train_dl_local, train_ds_local = get_dataloader(args, X_train[net_dataidx_map[client_idx]],
                                                            y_train[net_dataidx_map[client_idx]],
                                                            args.dataset, args.datadir, args.batch_size,
                                                            is_labeled=True,
                                                            data_idxs=net_dataidx_map[client_idx],
                                                            pre_sz=args.pre_sz, input_sz=args.input_sz)
            if args.dataset == 'skin':
                w, loss, op = local.train(args, sup_net_locals[client_idx].state_dict(), optimizer,
                                          train_dl_local, n_classes, X_test=X_test, y_test=y_test, res=True)  # network, loss, optimizer
            else:
                w, loss, op = local.train(args, sup_net_locals[client_idx].state_dict(), optimizer,
                                          train_dl_local, n_classes, res=True)

            local_w.append(w)
            sup_optim_locals[client_idx] = copy.deepcopy(op)
            if args.dataset == 'skin':
                local_num.append(len(net_dataidx_map[client_idx])*sc)
            else:
                local_num.append(len(net_dataidx_map[client_idx]))

        for client_idx in unsupervised_user_id:
            local = pl_trainer_locals[client_idx - sup_num]
            optimizer = pl_optim_locals[client_idx - sup_num]
            train_dl_local, train_ds_local = get_dataloader(args, X_train[net_dataidx_map[client_idx]],
                                                            y_train[net_dataidx_map[client_idx]],
                                                            args.dataset, args.datadir, args.batch_size,
                                                            is_labeled=False,
                                                            data_idxs=net_dataidx_map[client_idx],
                                                            pre_sz=args.pre_sz, input_sz=args.input_sz)
            w, op, num, train_label= local.train(args, pl_net_locals[client_idx - sup_num].state_dict(), optimizer,
                                      train_dl_local, n_classes, is_train=True, class_confident=class_confident, include_second=include, second_c=second_class, second_h=second_h)  # network, loss, optimizer
            local_w.append(w)
            pl_optim_locals[client_idx - sup_num] = copy.deepcopy(op)
            local_num.append(num)
            local_label = local_label + train_label

        local_label = local_label + sup_label
        print(local_label)
        local_label = (local_label / sum(local_label))*(n_classes/10)
        second_class = []
        second_h = []
        for i in range(len(local_label)):
            if(local_label[i]<T_lower):
                second_class.append(i)
            if(local_label[i]>T_higher):
                second_h.append(i)
        print(local_label)

        if min(local_label)<T_lower:
            include = True
        else:
            include = False
        local_label = local_label
        class_confident = local_label + T_base - local_label.std()
        if args.dataset == 'skin' or args.dataset == 'SVHN':
            class_confident[class_confident >= 0.9] = 0.9
        else:
            class_confident[class_confident >= T_upper] = T_upper

        w = FedAvg(net_glob.state_dict(), local_w, local_num)
        if args.dataset == 'skin':
            if com_round>0 and com_round%5==0:
                print('res weight connection 5 epoch')
                w_l = [record_w, copy.deepcopy(w)]
                n_l = [1., 1.]
                print(n_l)
                w = FedAvg(record_w, w_l, n_l)
                record_w = copy.deepcopy(w)
        else:
            if com_round>0 and com_round%5==0:
                print('res weight connection 5 epoch')
                w_l = [record_w, copy.deepcopy(w)]
                n_l = [1., 1.]
                print(n_l)
                w = FedAvg(record_w, w_l, n_l)
                record_w = copy.deepcopy(w)


        net_glob.load_state_dict(w)
        AUROC_avg, Accus_avg = test(com_round, net_glob.state_dict(), X_test, y_test, n_classes)
        print(args.dataset)
        print(T_lower, T_base)
        print(f'scaling factor: {sc}')
        print(AUROC_avg, Accus_avg)
        print(args.base_lr, args.unsup_lr)
        record_accuracy.append(Accus_avg)
        #print('adding lambda')
        print(record_accuracy)
        for i in supervised_user_id:
            sup_net_locals[i].load_state_dict(w)
        for i in unsupervised_user_id:
            pl_net_locals[i - sup_num].load_state_dict(w)