from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from models import dynamicnet, StegaStampEncoder
import numpy as np
import os
import time
import torch
import torchvision
import config
import torch.nn.functional as F


class BackdoorDatasets(Dataset):

    def __init__(self, opt, dataset, mode="train", rate=0.1, transform=None):
        # self.device = opt.device
        self.opt = opt
        self.dataname = opt.dataset
        self.transform = transform
        self.mode = mode

        if 'cifar' in self.dataname:
            self.opt.input_channel = 3
            self.opt.input_width = 32
            self.opt.input_height = 32
        else:
            raise Exception("Invalid dataset!")

        if opt.trigger_type == 'dynamicTrigger':
            if self.dataname == 'cifar10' or self.dataname == 'cifar100':
                model_path = opt.dynamic_model_path
            device = torch.device('cuda')
            state_dict = torch.load(model_path, map_location=device)
            self.netG = dynamicnet.Generator(self.opt).to(device)
            self.netG.load_state_dict(state_dict['netG'])
            self.netM = dynamicnet.Generator(self.opt, out_channels=1).to(device)
            self.netM.load_state_dict(state_dict['netM'])
            self.netG.eval()
            self.netM.eval()
        elif opt.trigger_type == 'blendHelloKitty':
            mask_path = os.path.join('trigger/hello_kitty_' + str(self.opt.input_height) + '.npy')
            self.mask = np.load(mask_path)
        elif opt.trigger_type == 'blendRandom':
            mask_path = os.path.join('trigger/blend_signal_'+str(self.opt.input_height)+'.npy')
            self.mask = np.load(mask_path)
        elif opt.trigger_type == 'ISSBA':
            state_dict = torch.load('trigger/ISSBA_cifar10.pth')
            self.secret = torch.load('trigger/secret').cuda()
            self.encoder = StegaStampEncoder(secret_size=len(self.secret),
                                             height=self.opt.input_height, width=self.opt.input_width, in_channel=3).cuda()
            self.encoder.load_state_dict(state_dict['encoder_state_dict'])
        elif opt.trigger_type == 'signalTrigger' and dataset == 'tiny-imagenet':
            self.mask = np.zeros([64, 64], dtype=float)
            for ix in range(64):
                for jx in range(64):
                    self.mask[ix, jx] = 30 / 255 * np.sin(2 * np.pi * jx * 6 / 64)

        self.dataset = self.add_trigger(dataset,
                                        opt.target_label,
                                        opt.target_type,
                                        opt.trigger_type,
                                        rate,
                                        mode)

    def __getitem__(self, item):
        img = self.dataset[item][0]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        label_idx = self.dataset[item][1]
        gt_label = self.dataset[item][2]
        is_bd = self.dataset[item][3]

        return img, label_idx, gt_label, is_bd

    def __len__(self):
        return len(self.dataset)

    def add_trigger(self, dataset, target_label, target_type, trigger_type, portion, mode):
        print("## using " + target_type + " attack")
        print("## generate " + mode + " Bad Imgs")
        total_idx = np.random.permutation(len(dataset))
        perm = total_idx[0: int(len(dataset) * portion)]
        if mode == 'mixed':
            perm = perm[: int(len(perm) / 3)]
        perm_rg = total_idx[len(perm): int(len(dataset) * portion)]
        #  new datasets
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            #  all to one attack
            if target_type == 'all_to_one':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        # select trigger
                        img = self.select_trigger(img, width, height, trigger_type)

                        # change target
                        # (image, target_label, ground-truth label, is bad?)
                        dataset_.append((img, target_label, data[1], True))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1], data[1], False))
                elif mode == 'mixed':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        # select trigger
                        img = self.select_trigger(img, width, height, trigger_type)

                        # change target
                        # (image, target_label, ground-truth label, is bad)
                        dataset_.append((img, target_label, data[1], True))
                        cnt += 1
                    elif i in perm_rg:
                        img = self.select_trigger(img, width, height, trigger_type)
                        dataset_.append((img, data[1], data[1], True))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1], data[1], False))
                else:
                    # if data[1] == target_label:
                    #     if portion == 0:
                    #         dataset_.append((np.array(data[0]), data[1]))
                    #     continue

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

                        dataset_.append((img, target_, data[1], True))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1], data[1], False))

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

                            dataset_.append((img, data[1], data[1], True))
                            cnt += 1

                        else:
                            dataset_.append((img, data[1], data[1], False))
                    else:
                        dataset_.append((img, data[1], data[1], False))

                else:
                    # if data[1] == target_label:
                    #     if portion == 0:
                    #         dataset_.append((np.array(data[0]), data[1]))
                    #     continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.select_trigger(img, width, height, trigger_type)

                        dataset_.append((img, target_label))
                        cnt += 1
                    else:
                        dataset_.append((img, data[1]))

        print("Injecting Over: " + str(cnt) + "Bad Imgs, " + str(len(dataset) - cnt) + "Clean Imgs")

        return dataset_

    def _change_label_next(self, label):
        if self.dataname == 'cifar10':
            nums = 10
        elif self.dataname == 'imagenet-subset':
            nums = 200
        else:
            raise Exception('Error dataset!')
        label_new = ((label + 1) % nums)
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
            img = self._blendRandomTrigger(img, width, height)

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

    def _blendRandomTrigger(self, img, width, height):
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

    def _ISSBA(self, img, width, height):
        img_ = torch.FloatTensor(img).div(255)
        img_ = img_.permute(2, 0, 1).unsqueeze(0)
        residual = self.encoder([self.secret, img_.cuda()]).detach().cpu()
        encoded_img = (img_ + residual).clamp(0, 1)
        img_ = (np.array(encoded_img[0].mul(255).byte(), dtype=np.uint8)).transpose(1, 2, 0)
        return img_


def trans(dataset):
    trans_list = [torchvision.transforms.ToTensor()]
    trans_list.append(torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]))
    return torchvision.transforms.Compose(trans_list)


def main():
    opt = config.get_arguments().parse_args()
    ori_datasets = {
        "cifar10": torchvision.datasets.CIFAR10(root=opt.dataset_path,
                                                train=True),
        "cifar100": torchvision.datasets.CIFAR100(root=opt.dataset_path,
                                                  train=True)
    }
    train_dataset = BackdoorDatasets(opt=opt,
                                     dataset=ori_datasets[opt.dataset],
                                     mode='train',
                                     rate=opt.poisoned_rate)
    data_path = "./poisoned_datasets/{}/{}-{}-{}.npy".format(opt.trigger_type, opt.dataset, opt.target_type,
                                                             opt.poisoned_rate)
    print('Dataset : {}'.format(len(train_dataset)))
    print('Start saving.')
    start = time.time()
    np.save(data_path, train_dataset)
    end = time.time()
    print('Use {} seconds.'.format((end - start)))


if __name__ == '__main__':
    main()
