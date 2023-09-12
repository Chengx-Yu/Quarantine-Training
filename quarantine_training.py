import torch
import torchvision
import time
import os
import random
import operator
import copy
import config
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import kornia.augmentation as K

from math import sqrt
from tqdm import tqdm
from PIL import Image
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

from models import PreActResNet18


def get_low_loss_idx(opt):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    split_model_path = f'checkpoints/pre-qt/{opt.target_type}-{opt.trigger_type}-{opt.use_model}-{opt.dataset}-' \
                       f'poison{opt.poisoned_rate}.path'
    model = {
        "PreActResNet18": PreActResNet18(num_classes=opt.num_classes),
        "ResNet18": torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1),
        "MobileNet": MobileNet32(num_classes=opt.num_classes),
        "ShuffleNetV2": ShuffleNetV2(net_size=0.5)
    }
    net = model[opt.use_model]
    if opt.use_model == 'ResNet18':
        fc_features = net.fc.in_features
        net.fc = torch.nn.Linear(fc_features, opt.num_classes)
        net = net.to(opt.device)
    net.load_state_dict(torch.load(split_model_path))
    net = net.to(opt.device)
    net.eval()

    train_data = np.load(opt.poisoned_dataset, allow_pickle=True)
    train_data = ReloadDataset(train_data, transform=get_trans(opt.dataset, False))
    each_dataloader = DataLoader(dataset=train_data, batch_size=1, shuffle=False)

    losses_record = []

    with torch.no_grad():
        for step, (img, target, _, is_bad, img_idx) in enumerate(tqdm(each_dataloader)):
            img, target = img.cuda(), target.cuda()

            output = net(img)
            loss = criterion(output, target)

            losses_record.append(loss.item())
    losses_idx = np.argsort(np.array(losses_record))

    neg_idx = torch.tensor(losses_idx[:500]).cuda()

    return neg_idx


def change_label(targets, nums_class, img_idx, qt_idx, neg_idx, epo=0):
    ori_targets = copy.deepcopy(targets)
    quarantine_targets = targets + nums_class
    # if epo == 0:
    #     new_targets = torch.zeros((targets.size(0), (nums_class * 2))).scatter_(1, targets.unsqueeze_(1), 0.55)
    #     new_targets = new_targets.scatter_(1, (targets + nums_class), 0.45)
    if epo < 6:
        ori = torch.zeros((targets.size(0), (nums_class * 2))).scatter_(1, targets.unsqueeze_(1), 1)
        qt = torch.zeros((targets.size(0), (nums_class * 2))).scatter_(1, (targets + nums_class), 1)
        for i, idx in enumerate(img_idx):
            if idx in qt_idx:
                ori[i] *= float(np.clip((0.6 - epo * 0.1), a_min=0.1, a_max=0.9))
                qt[i] *= float(np.clip((0.4 + epo * 0.1), a_min=0.1, a_max=0.9))
            else:
                ori[i] *= 0.6
                qt[i] *= 0.4
            if idx in neg_idx:
                ori[i] *= 0.1
                qt[i] *= 0.9
        new_targets = ori + qt
    else:
        ori = torch.zeros((targets.size(0), (nums_class * 2))).scatter_(1, targets.unsqueeze_(1), 1)
        qt = torch.zeros((targets.size(0), (nums_class * 2))).scatter_(1, (targets + nums_class), 1)
        for i, idx in enumerate(img_idx):
            if idx in qt_idx:
                ori[i] *= 0.1
                qt[i] *= 0.9
            else:
                ori[i] *= 0.9
                qt[i] *= 0.1
        new_targets = ori + qt

    return new_targets, ori_targets, quarantine_targets


def free_label(qt_idx, dataloader, opt):
    qt_idx_free = []
    qt_class_cnt = np.array([0] * opt.num_classes)
    total_class_cnt = np.array([0] * opt.num_classes)
    for step, (inputs, targets, _, is_bad, img_idx) in enumerate(tqdm(dataloader)):
        targets, img_idx = targets.cuda(), img_idx.cuda()
        for t in targets:
            total_class_cnt[t] += 1
        for c, i in enumerate(img_idx):
            if i in qt_idx:
                qt_class_cnt[targets[c]] += 1
    # max_class, max_number = max(enumerate(qt_class_cnt), key=operator.itemgetter(1))
    wrong_class = np.where(qt_class_cnt >= total_class_cnt * 0.9)
    for c, i in enumerate(qt_class_cnt):
        print("class {} has {} images in qt.".format(c, i))
    if len(wrong_class) > 0:
        print("Free qt class {}".format(wrong_class[0]))
        for step, (inputs, targets, _, is_bad, img_idx) in enumerate(tqdm(dataloader)):
            for c, i in enumerate(img_idx):
                if i in qt_idx and targets[c].item() not in wrong_class[0]:
                    qt_idx_free.append(i)
    return torch.tensor(qt_idx_free).cuda()

class BackdoorDatasets(Dataset):

    def __init__(self, opt, dataset, mode="train", portion=0.1, transform=None):
        self.device = opt.device
        self.dataname = opt.dataset
        self.transform = transform
        self.opt = opt
        self.mode = mode
        if 'cifar' in self.dataname:
            self.opt.input_channel = 3
            self.opt.input_width = 32
            self.opt.input_height = 32
        else:
            raise Exception("Invalid dataset!")
        # self.classes = datasets.classes
        # self.class_to_idx = datasets.class_to_idx
        if opt.trigger_type == 'blendHelloKitty':
            mask_path = os.path.join('trigger/hello_kitty_' + str(self.opt.input_height) + '.npy')
            self.mask = np.load(mask_path)
        elif opt.trigger_type == 'blendRandom':
            mask_path = os.path.join('trigger/blend_signal_' + str(self.opt.input_height) + '.npy')
            self.mask = np.load(mask_path)
        self.dataset = self.add_trigger(dataset, opt.target_label, opt.target_type, opt.trigger_type, portion, mode)

    def __getitem__(self, item):
        img = self.dataset[item][0]
        if self.transform is not None:
            img = self.transform(img)
        img = img.to(self.device)

        label_idx = self.dataset[item][1]
        if self.dataname == 'cifar100':
            class_num = 100
        else:
            class_num = 10
        label = np.zeros(class_num)
        label[label_idx] = 1
        label = torch.Tensor(label)
        label = label.to(self.device)

        return img, label

    def __len__(self):
        return len(self.dataset)

    def add_trigger(self, dataset, target_label, target_type, trigger_type, portion, mode):
        print("## using " + target_type + " attack")
        print("## generate " + mode + " Bad Imgs")
        total_idx = np.random.permutation(len(dataset))
        perm = total_idx[0: int(len(dataset) * portion)]
        #  new datasets
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            #  all to one attack
            if target_type == 'all_to_one':

                if mode == "train":
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        # select trigger
                        img = self.select_trigger(img, width, height, trigger_type)

                        # change target
                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))
                else:
                    # if data[1] == target_label:
                    if portion == 0:
                        dataset_.append((np.array(data[0]), data[1]))
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.select_trigger(img, width, height, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # all2all attack
            elif target_type == 'all_to_all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:

                        img = self.select_trigger(img, width, height, trigger_type)
                        target_ = self._change_label_next(data[1])

                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

                else:
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.select_trigger(img, width, height, trigger_type)

                        target_ = self._change_label_next(data[1])
                        dataset_.append((img, target_))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

            # clean label attack
            elif target_type == 'clean_label':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.select_trigger(img, width, height, trigger_type)

                            dataset_.append((img, data[1]))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1]))
                    else:
                        dataset_.append((img, data[1]))

                else:
                    # if data[1] == target_label:
                    if portion == 0:
                        dataset_.append((np.array(data[0]), data[1]))
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.select_trigger(img, width, height, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        time.sleep(0.01)
        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")

        return dataset_

    def _change_label_next(self, label):
        label_new = ((label + 1) % 10)
        return label_new

    def select_trigger(self, img, width, height, triggerType):

        assert triggerType in ['badnetTrigger', 'gridTrigger', 'fourCornerTrigger', 'blendRandom',
                               'signalTrigger', 'trojanTrigger', 'dynamicTrigger',
                               'blendHelloKitty', 'ISSBA']

        if triggerType == 'badnetTrigger':
            img = self._badnetTrigger(img, width, height)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height)

        elif triggerType == 'blendRandom':
            img = self._randomPixelTrigger(img, width, height)

        elif triggerType == 'blendHelloKitty':
            img = self._blendHelloKitty(img, width, height)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height)

        elif triggerType == 'dynamicTrigger':
            img = self._dynamicTrigger(img, width, height)

        elif triggerType == 'ISSBA':
            img = self._ISSBA(img, width, height)

        else:
            raise NotImplementedError

        return img

    def _badnetTrigger(self, img, width, height):
        img[width - 3][height - 3] = 255
        img[width - 3][height - 2] = 0
        img[width - 2][height - 3] = 0
        img[width - 2][height - 2] = 255

        return img

    def _gridTriger(self, img, width, height):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        return img

    def _fourCornerTrigger(self, img, width, height):
        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height):
        alpha = 0.2
        # mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * self.mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _blendHelloKitty(self, img, width, height):
        alpha = 0.2
        blend_img = (1 - alpha) * img + alpha * self.mask
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _signalTrigger(self, img, width, height):
        alpha = 0.2
        # load signal mask
        signal_mask = np.load('./trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height):
        # load trojanmask
        trg = np.load('./trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        return img_


class ReloadDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.transform = transform
        self.dataset = self.add_flag(dataset)

    def __getitem__(self, item):
        img = self.dataset[item][0]
        if self.transform is not None:
            img = self.transform(img)
        label_idx = self.dataset[item][1]
        gt_label = self.dataset[item][2]
        is_bad = self.dataset[item][3]
        img_idx = self.dataset[item][4]
        return img, label_idx, gt_label, is_bad, img_idx

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def add_flag(dataset):
        new_set = []
        for i, data in enumerate(dataset):
            new_set.append((data[0], data[1], data[2], data[3], i))
        return new_set


def get_trans(data_name, is_train):
    trans_list = [transforms.ToTensor()]
    if data_name == 'cifar10' or data_name == 'cifar100':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    else:
        raise Exception('Invalid dataset!')
    trans_list.append(normalize)
    return transforms.Compose(trans_list)


def load_dataloader(opt):
    if opt.dataset == 'cifar10':
        test_data = datasets.CIFAR10(root=opt.dataset_path, train=False)
    elif opt.dataset == 'cifar100':
        test_data = datasets.CIFAR100(root=opt.dataset_path, train=False)

    train_data0 = np.load(opt.poisoned_dataset, allow_pickle=True)
    train_data = ReloadDataset(train_data0, transform=get_trans(opt.dataset, False))
    test_data_cl = BackdoorDatasets(opt=opt,
                                    dataset=test_data,
                                    mode='test',
                                    portion=0,
                                    transform=get_trans(opt.dataset, False))
    test_data_bd = BackdoorDatasets(opt=opt,
                                    dataset=test_data,
                                    mode='test',
                                    portion=1,
                                    transform=get_trans(opt.dataset, False))

    train_data_loader = DataLoader(dataset=train_data, batch_size=opt.bs, shuffle=True)
    test_data_loader_cl = DataLoader(dataset=test_data_cl, batch_size=opt.bs, shuffle=False)
    test_data_loader_bd = DataLoader(dataset=test_data_bd, batch_size=opt.bs, shuffle=False)

    return train_data_loader, test_data_loader_cl, test_data_loader_bd


def net_prepare(opt):
    model = {
        "PreActResNet18": PreActResNet18(num_classes=opt.num_classes)
    }
    net = model[opt.use_model]
    split_model_path = f'checkpoints/filter/split/{opt.target_type}-{opt.trigger_type}-{opt.use_model}-{opt.dataset}-' \
                       f'poison{opt.poisoned_rate}.path'
    net.load_state_dict(torch.load(split_model_path))
    fc_features = net.linear.in_features
    net.linear = torch.nn.Linear(fc_features, opt.num_classes * 2)
    optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100)

    return net.to(opt.device), optimizer, scheduler


class PostTensorTransform(torch.nn.Module):
    def __init__(self):
        super(PostTensorTransform, self).__init__()
        self.random_crop = K.RandomCrop((32, 32), padding=4)
        self.rotation = K.RandomHorizontalFlip(p=0.5)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


def train_process(dataloader, net, optimizer, scheduler, qt_idx=torch.tensor([]), neg_idx=torch.tensor([]), epo=0):
    running_loss = 0.0
    total_sample = 0
    ori_correct = 0
    clean_qt_correct = 0
    trojan_qt_correct = 0
    clean_correct = 0
    trojan_correct = 0
    clean_nums = 0
    trojan_nums = 0

    qt_idx_new = torch.tensor([], dtype=torch.long).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    trans = PostTensorTransform().cuda()
    net.train()

    for step, (inputs, targets, _, is_bad, img_idx) in enumerate(tqdm(dataloader)):
        new_targets, ori_targets, qt_targets = change_label(targets, 10, img_idx, qt_idx, neg_idx, epo)
        optimizer.zero_grad()
        bs = inputs.shape[0]
        total_sample += bs
        inputs, new_targets, ori_targets, qt_targets = inputs.cuda(), new_targets.cuda(), ori_targets.cuda(), qt_targets.cuda()
        is_bad, img_idx = is_bad.cuda(), img_idx.cuda()
        clean_idx = torch.where(is_bad == False)
        trojan_idx = torch.where(is_bad == True)
        clean_nums += len(targets[clean_idx])
        trojan_nums += len(targets[trojan_idx])
        inputs = inputs.type(torch.cuda.FloatTensor)

        if epo > 19:
            inputs = trans(inputs)
        output = net(inputs)
        predict = torch.argmax(output, dim=1)

        qt_idx_new = torch.cat((qt_idx_new, img_idx[torch.where(predict == qt_targets)]))

        ori_correct += torch.sum(predict == ori_targets)
        clean_qt_correct += torch.sum(predict[clean_idx] == qt_targets[clean_idx])
        trojan_qt_correct += torch.sum(predict[trojan_idx] == qt_targets[trojan_idx])
        clean_correct += torch.sum(predict[clean_idx] == ori_targets[clean_idx])
        trojan_correct += torch.sum(predict[trojan_idx] == ori_targets[trojan_idx])

        loss = criterion(output, new_targets)

        running_loss += loss
        loss.backward()

        optimizer.step()
        scheduler.step()
    acc_train_ori = ori_correct / total_sample * 100.
    acc_clean_qt = clean_qt_correct / clean_nums * 100.
    acc_trojan_qt = trojan_qt_correct / trojan_nums * 100.
    acc_clean = clean_correct / clean_nums * 100.
    acc_trojan = trojan_correct / trojan_nums * 100.

    if epo > 19:
        qt_idx_new = qt_idx

    return running_loss.item(), acc_train_ori.item(), acc_clean_qt.item(), \
           acc_trojan_qt.item(), acc_clean.item(), acc_trojan.item(), qt_idx_new


def eval_process(dataloader, net):
    net.eval()
    total_sample = 0
    correct = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(tqdm(dataloader)):
            bs = inputs.shape[0]
            total_sample += bs
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs = inputs.type(torch.cuda.FloatTensor)
            output = net(inputs)

            predict = torch.argmax(output, dim=1)
            truth = torch.argmax(targets, dim=1)
            correct += torch.sum(predict == truth)

        acc_test = correct / total_sample * 100.

    return acc_test.item()


def main():
    opt = config.get_arguments().parse_args()
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_file = f'./checkpoints/quarantine'
    if not os.path.exists(model_file):
        os.makedirs(model_file)
    opt.model_path = f'./checkpoints/quarantine/{opt.target_type}-{opt.trigger_type}-{opt.use_model}-{opt.dataset}-' \
                     f'poison{opt.poisoned_rate}.path'
    opt.poisoned_dataset = "./poisoned_datasets/{}/{}-{}-{}.npy".format(opt.trigger_type, opt.dataset, opt.target_type,
                                                                        opt.poisoned_rate)

    if opt.dataset == 'cifar10':
        opt.num_classes = 10
        opt.input_width = 32
        opt.input_height = 32
    elif opt.dataset == 'cifar100':
        opt.num_classes = 100
        opt.input_width = 32
        opt.input_height = 32

    net, optimizer, scheduler = net_prepare(opt=opt)

    train_data_loader, test_data_loader_cl, test_data_loader_bd = load_dataloader(opt=opt)

    best_acc = 0.0
    best_asr = 0.0
    doc = []

    if opt.continue_training:
        print("Continue training!")
        state_dict = torch.load(opt.model_path)
        net.load_state_dict(state_dict["net"])
        start_epo = state_dict["epoch_current"]
        best_acc = state_dict["best_acc"]
        best_asr = state_dict["best_asr"]
        qt_idx = state_dict["qt_idx"]
        qt_idx = qt_idx.to(opt.device)
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
    else:
        start_epo = 0
        qt_idx = torch.tensor([], dtype=torch.long).to(opt.device)
    neg_idx = get_low_loss_idx(opt)
    # neg_idx = torch.tensor([])

    opt.epoch = 10
    for epo in range(start_epo, opt.epoch):
        loss, acc_train_ori, acc_clean_qt, acc_trojan_qt, acc_clean, acc_trojan, qt_idx = \
            train_process(dataloader=train_data_loader,
                          net=net,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          qt_idx=qt_idx,
                          neg_idx=neg_idx,
                          epo=epo)
        acc_test = eval_process(dataloader=test_data_loader_cl, net=net)
        asr_test = eval_process(dataloader=test_data_loader_bd, net=net)
        if epo in [5, 9, 14, 19]:
            qt_idx = free_label(qt_idx, train_data_loader, opt)
          
        if acc_test > best_acc:
            print("Saving...")
            best_acc = acc_test
            best_asr = asr_test
            state_dict = {
                "net": net.state_dict(),
                "scheduler": scheduler.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_acc": best_acc,
                "best_asr": best_asr,
                "epoch_current": epo,
                "qt_idx": qt_idx
            }
            torch.save(state_dict, opt.model_path)

        print('train loss : {:.2f} | train acc : {:.2f} | '
              'clean quarantine acc : {:.2f} | train clean acc : {:.2f} | '
              'trojan quarantine acc : {:.2f} | train trojan acc : {:.2f} | Epoch : {:}\n'
              'test clean : {:.2f} | test bad : {:.2f}\n'
              'best acc : {:.2f} | best asr : {:.2f}'
              .format(loss, acc_train_ori, acc_clean_qt, acc_clean, acc_trojan_qt, acc_trojan, epo, acc_test, asr_test, best_acc, best_asr))

        if not epo % 1:
            doc.append(
                (epo + 1, acc_train_ori, acc_test, asr_test, best_acc, best_asr, acc_clean_qt, acc_trojan_qt))
            df = pd.DataFrame(doc, columns=('epoch', 'train_acc', 'test_acc', 'test_asr', 'best_acc', 'best_asr', 'clean_qt', 'trojan_qt'))
            # df.to_csv(f"./log/quarantine/{opt.use_model}-{opt.dataset}-{opt.target_type}-{opt.trigger_type}"
            #           f"-poison{opt.poisoned_rate}.csv",
            #           index=False, encoding='utf-8')
            df.to_csv(f"./log/{opt.use_model}-{opt.dataset}-{opt.target_type}-{opt.trigger_type}"
                      f"-poison{opt.poisoned_rate}.csv",
                      index=False, encoding='utf-8')


if __name__ == "__main__":
    main()

