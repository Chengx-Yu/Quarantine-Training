import torch
import torchvision
import config
import copy
import os
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from data import TransformedDataset
from models import PreActResNet18

def trans(dataset):
    trans_list = [torchvision.transforms.ToTensor()]
    if dataset == 'cifar10' or dataset == 'cifar100':
        trans_list.append(torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]))
    elif dataset == 'imagenet-subset':
        trans_list.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    return torchvision.transforms.Compose(trans_list)


def load_dataloader(opt):

    train_data = np.load(opt.poisoned_dataset, allow_pickle=True)
    train_data_copy = copy.deepcopy(train_data)
    train_data = TransformedDataset(train_data, transform=trans(opt.dataset))
    train_data_loader = DataLoader(dataset=train_data, batch_size=opt.bs, shuffle=True)
    train_data_loader_order = DataLoader(dataset=train_data, batch_size=opt.bs, shuffle=False)

    return train_data_loader, train_data_loader_order, train_data_copy


def net_prepare(opt):
    if opt.use_model == 'PreActResNet18':
        net = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    else:
        raise Exception('Error model!')

    optimizer = torch.optim.SGD(net.parameters(), opt.lr, momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100)

    return net, optimizer, scheduler

def easy_train(net, optimizer, scheduler, train_data_loader, opt):
    total_clean_correct = 0
    total_bd_correct = 0
    num_clean = 0
    num_bd = 0
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for step, (inputs, targets, gt_labels, is_bad) in enumerate(tqdm(train_data_loader, desc='Training')):
        optimizer.zero_grad()
        inputs, targets, gt_labels, is_bad = inputs.to(opt.device), targets.to(opt.device), gt_labels.to(
            opt.device), is_bad.to(opt.device)
        clean_idx, trojan_idx = torch.where(is_bad == False), torch.where(is_bad == True)

        for v in is_bad:
            if v:
                num_bd += 1
            elif not v:
                num_clean += 1

        output = net(inputs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        total_clean_correct += torch.sum(torch.argmax(output[clean_idx], dim=1) == targets[clean_idx])
        total_bd_correct += torch.sum(torch.argmax(output[trojan_idx], dim=1) == targets[trojan_idx])
    clean_acc = total_clean_correct * 100. / num_clean
    trojan_acc = total_bd_correct * 100. / num_bd
    print("Clean ACC : {:.4f} | Trojan ACC : {:.4f}".format(clean_acc.item(), trojan_acc.item()))

    scheduler.step()

def main():
    opt = config.get_arguments().parse_args()
    opt.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_file = f'./checkpoints/pre-qt'
    if not os.path.exists(model_file):
        os.makedirs(model_file)
    opt.model_path = f'./checkpoints/pre-qt/{opt.target_type}-{opt.trigger_type}-{opt.use_model}-{opt.dataset}-' \
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

    train_data_loader, train_data_loader_order, train_data = load_dataloader(opt=opt)
    for epo in range(4):
        easy_train(net, optimizer, scheduler, train_data_loader, opt)
        torch.save(net.state_dict(), opt.model_path)
if __name__ == "__main__":
    main()
