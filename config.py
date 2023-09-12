import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description='Backdoor setting.')

    # ------------------------- training setting -------------------------
    parser.add_argument('--dataset', default='cifar10')
    parser.add_argument('--use_model', default='PreActResNet18')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dataset_path', default='./datasets')
    parser.add_argument('--continue_training', action='store_true')

    # -------------------------- attack setting ---------------------------
    parser.add_argument('--target_type', default='all_to_one')
    parser.add_argument('--trigger_type', default='badnetTrigger')
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--poisoned_rate', type=float, default=0.1)

    return parser
