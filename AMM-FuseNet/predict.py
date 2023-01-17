from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np
from einops import rearrange
from torch.utils import data

from datasets import Hunan_dual_2D
from datasets import Hunan2_dual_2D

from datasets import potsdam_dual_2D
# from datasets import vaihingen_dual_2D

from datasets import DFC20_dual_2D
from utils import ext_transforms as et
from metrics import StreamSegMetrics
import torch
import torch.nn as nn
from utils.visualizer import Visualizer
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as T
import kornia.augmentation as K

def get_argparser():
    parser = argparse.ArgumentParser()
    # Datset Options
    parser.add_argument("--data_root", type=str, default="./data/myhunan",
                        help="path to Dataset")
    # potsdam       dfc20       hunan    hunan2
    parser.add_argument("--dataset", type=str, default='hunan',
                        choices=['potsdam',  'hunan', 'hunan2', 'dfc20'], help='Name of dataset')

    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    #     dual_parasingle_nopretrained
    #     dual_parasingle_pretrained


    parser.add_argument("--model", type=str, default='dual_parasingle_nopretrained',
                        choices=available_models, help='model name')

    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    # Train Options
    parser.add_argument("--test_only", action='store_true', default=True)
    parser.add_argument("--save_val_results", action='store_true', default=True,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--val_batch_size", type=int, default=1,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='8097',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser

def get_dataset(opts):

    if opts.dataset == 'hunan':
        def preprocess(sample):
            for i in range(13):
                sample["modality1"][i] = (sample["modality1"][i] - torch.min(sample["modality1"][i])) / (
                            torch.max(sample["modality1"][i]) - torch.min(sample["modality1"][i]))
            for i in range(2):
                sample["modality2"][i] = (sample["modality2"][i] - torch.min(sample["modality2"][i])) / (
                            torch.max(sample["modality2"][i]) - torch.min(sample["modality2"][i]))
            return sample
        transforms = T.Compose([preprocess])
        test_dataset = Hunan_dual_2D(root=opts.data_root, split="test", transforms=transforms)
    elif opts.dataset == 'hunan2':
        def preprocess(sample):
            for i in range(13):
                sample["modality1"][i] = (sample["modality1"][i] - torch.min(sample["modality1"][i])) / (
                            torch.max(sample["modality1"][i]) - torch.min(sample["modality1"][i]))
                # sample["image"][i] = torch.clip(sample["image"][i], min=0.0, max=1.0)
            #normalise
            max = 1892.0 
            min = 18.0
            sample["modality2"] = (sample["modality2"] - min) / (max - min)
            sample["modality2"] = torch.clip(sample["modality2"], min=0.0, max=1.0)
            return sample
        transforms = T.Compose([preprocess])
        test_dataset = Hunan2_dual_2D(root=opts.data_root, split="test", transforms=transforms)

    elif opts.dataset == 'potsdam':

        def preprocess(sample):
            sample["modality1"][:] /= 255.0
            sample["modality2"] = (sample["modality2"] - torch.min(sample["modality2"])) / (
                    torch.max(sample["modality2"]) - torch.min(sample["modality2"]))
            return sample

        transforms = T.Compose([preprocess])
        test_dataset = potsdam_dual_2D(root=opts.data_root, split="test", transforms=transforms)


    elif opts.dataset == 'dfc20':
        test_dataset = DFC20_dual_2D(root=opts.data_root, split="test")


    else:
        raise RuntimeError("Dataset not found")
    return test_dataset


def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')
        img_id = 0

    with torch.no_grad():
        for i, sample in tqdm(enumerate(loader)):
            labels = sample['mask'].to(device, dtype=torch.long)
            modality1 = sample['modality1'].to(device, dtype=torch.float32)
            modality2 = sample['modality2'].to(device, dtype=torch.float32)

            outputs = model(modality1, modality2)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (modality1[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results and i%50==0:
                for i in range(len(modality1)):
                    image = modality1[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (image[1:4] * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = loader.dataset.decode_target(target).astype(np.uint8)
                    pred = loader.dataset.decode_target(pred).astype(np.uint8)

                    Image.fromarray(image).save('results/%d_image.png' % img_id)
                    Image.fromarray(target).save('results/%d_target.png' % img_id)
                    Image.fromarray(pred).save('results/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    print("Running main")
    
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'hunan' or opts.dataset.lower() == 'hunan2' :
        opts.num_classes = 7

    elif opts.dataset.lower() == 'potsdam' :
        opts.num_classes = 6

    elif opts.dataset.lower() == 'dfc20':
        opts.num_classes = 10
    else:
        raise RuntimeError("Dataset not found")

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    test_dataset = get_dataset(opts)
    val_loader = data.DataLoader(
        test_dataset, batch_size=opts.val_batch_size, shuffle=False, num_workers=2, drop_last=True)
    print("loaded dataset")

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](dataset=opts.dataset,num_classes=opts.num_classes, output_stride=opts.output_stride)


    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    if opts.ckpt is None:
        opts.ckpt = "checkpoints/latest_"+opts.model+"_"+opts.dataset+"_os"+str(opts.output_stride)+".pth"
    
    print(opts)
    
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("model not restored")
        return

    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

if __name__ == '__main__':
    main()
