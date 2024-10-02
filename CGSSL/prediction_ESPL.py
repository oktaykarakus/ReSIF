from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from einops import rearrange
from torch.utils import data
from datasets import Potsdam2D
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
import torchvision.transforms as T
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import kornia.augmentation as K
import psutil
import shutil
from tensorboardX import SummaryWriter
from utils import losses, ramps
import torchvision
from sklearn.metrics import normalized_mutual_info_score
from torch import Tensor
from torch.distributions import Categorical

def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="path to dataset",
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='potsdam',
                        choices=['potsdam'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name]))
    parser.add_argument('--in_number', type=int,  default=None,
                    help='input channel of network')
#-------------------------------------training setting----------------------------------------------
  
#----------------------unet  deeplabv3plus  pspnet   segnet
    parser.add_argument("--num_tra_sam", type=str, default='2downsample_train')  
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--batch_size", type=int, default=5,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=5,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--val_interval", type=int, default=50,
                        help='each 100 iteration, validate the model')
    parser.add_argument("--ckpt1", default="path to model1", type=str,
                        help="restore from checkpoint")
    parser.add_argument("--ckpt2", default="path to model2", type=str,
                        help="restore from checkpoint")
    parser.add_argument("--ckpt3", default="path to model3", type=str,
                        help="restore from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")   
    parser.add_argument('--device', type=str,  default='cuda',
                    help='computing device')
    # costs
    return parser

def get_dataset(opts):
    if opts.dataset == 'potsdam':
        MAX = 106.171
        MIN = -35.75
        def preprocess(sample):
            sample["image"][:4] /= 255.0
            sample["image"][4] = (sample["image"][4] - MIN) / (MAX - MIN)
            return sample

        transforms = torchvision.transforms.Compose([preprocess])
        db_test = Potsdam2D(root=opts.data_root, num_tra_sam=opts.num_tra_sam, split="test", transforms=transforms)
    else:
        raise RuntimeError("Dataset not found")
    return db_test

def validate(opts, model1, model2, model3, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    MAE = nn.L1Loss()
    average_loss = 0.0
    number = len(loader)
    with torch.no_grad():
        for i, sample in tqdm(enumerate(loader)):

            images = sample['image'].to(device, dtype=torch.float32)
            labels = sample['label'].to(device, dtype=torch.long)
            outputs1 = torch.softmax(model1(images),dim=1)
            outputs2 = torch.softmax(model2(images),dim=1)
            outputs3 = torch.softmax(model3(images),dim=1)
            outputs = outputs1 + outputs2 + outputs3

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
        score = metrics.get_results()
    return score



def main():
    opts = get_argparser().parse_args()
    
    if opts.dataset.lower() == 'potsdam':
        opts.in_number = 5
        opts.num_classes = 6
    else:
        raise RuntimeError("Dataset not found")

    # Setup visualization
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloaderloss_sup_2
    db_test = get_dataset(opts)
    test_loader = data.DataLoader(
        db_test, batch_size=opts.val_batch_size, shuffle=False, num_workers=0, drop_last=True)

    print("Dataset: %s, test set: %d" %
          (opts.dataset, len(db_test)))

    # Set up model (all models are 'constructed at network.modeling)
    # model1 = create_model(opts, ema=False) #unet
    model1 = network.modeling.__dict__['unet'](in_number=opts.in_number, num_classes=opts.num_classes)
    model2 = network.modeling.__dict__['segnet'](in_number=opts.in_number, num_classes=opts.num_classes)
    model3 = network.modeling.__dict__['pspnet'](in_number=opts.in_number, num_classes=opts.num_classes)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    if opts.ckpt1 is not None and opts.ckpt2 is not None and os.path.isfile(opts.ckpt1):
        checkpoint1 = torch.load(opts.ckpt1, map_location=torch.device('cpu'))
        checkpoint2 = torch.load(opts.ckpt2, map_location=torch.device('cpu'))
        checkpoint3 = torch.load(opts.ckpt3, map_location=torch.device('cpu'))
        model1.load_state_dict(checkpoint1["model_state"])
        model2.load_state_dict(checkpoint2["model_state"])
        model3.load_state_dict(checkpoint3["model_state"])
        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)
        model3 = nn.DataParallel(model3)
        model1.to(device)
        model2.to(device)
        model3.to(device)
        print("Model restored from %s" % opts.ckpt1)
        print("Model restored from %s" % opts.ckpt2)
        print("Model restored from %s" % opts.ckpt3)
        del checkpoint1  # free memory
        del checkpoint2  # free memory
        del checkpoint3  # free memory     
    else:
        raise RuntimeError("checkpoint not found")    

    model1.eval()
    model2.eval()
    model3.eval()
    val_score = validate(
        opts=opts, model1=model1, model2=model2, model3=model3, loader=test_loader, device=device, metrics=metrics)
    print(metrics.to_str(val_score))
               
if __name__ == '__main__':
    main()
