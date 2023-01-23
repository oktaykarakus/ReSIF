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

from datasets import DFC20_dual_2D

from datasets import passau
from datasets import Hunan3

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


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default="./data/myhunan",
                        help="path to Dataset")
    # potsdam     dfc20    hunan     hunan2
    parser.add_argument("--dataset", type=str, default='hunan',
                        choices=['potsdam', 'dfc20', 'hunan', 'hunan2', 'passau', 'hunan3'], help='Name of dataset')

    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None)")
    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower()
                              and not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='dual_parasingle_nopretrained',
                        choices=available_models, help='model name')

    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    parser.add_argument("--save_val_results", action='store_true', default=False,
                        help="save segmentation results to \"./results\"")
    parser.add_argument("--total_itrs", type=int, default=50e3,
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--batch_size", type=int, default=10,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=10,
                        help='batch size for validation (default: 4)')

    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

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
        train_dataset = Hunan_dual_2D(root=opts.data_root, split="train", transforms=transforms)
        val_dataset = Hunan_dual_2D(root=opts.data_root, split="val", transforms=transforms)

    elif opts.dataset == 'hunan2':
        def preprocess(sample):
            for i in range(13):
                sample["modality1"][i] = (sample["modality1"][i] - torch.min(sample["modality1"][i])) / (
                    torch.max(sample["modality1"][i]) - torch.min(sample["modality1"][i]))
            max = 1892.0
            min = 18.0
            sample["modality2"] = (sample["modality2"] - min) / (max - min)
            sample["modality2"] = torch.clip(sample["modality2"], min=0.0, max=1.0)
            return sample
        transforms = T.Compose([preprocess])
        train_dataset = Hunan2_dual_2D(root=opts.data_root, split="train", transforms=transforms)
        val_dataset = Hunan2_dual_2D(root=opts.data_root, split="val", transforms=transforms)

    elif opts.dataset == 'potsdam':

        def preprocess(sample):
            sample["modality1"][:] /= 255.0
            sample["modality2"] = (sample["modality2"] - torch.min(sample["modality2"])) / (
                torch.max(sample["modality2"]) - torch.min(sample["modality2"]))
            return sample

        transforms = T.Compose([preprocess])
        train_dataset = potsdam_dual_2D(root=opts.data_root, split="train", transforms=transforms)
        val_dataset = potsdam_dual_2D(root=opts.data_root, split="val", transforms=transforms)

    elif opts.dataset == 'dfc20':
        train_dataset = DFC20_dual_2D(root=opts.data_root, split="train")
        val_dataset = DFC20_dual_2D(root=opts.data_root, split="val")

    elif opts.dataset == 'passau':
        pass
        # TODO: dataset preprocessing

    elif opts.dataset == 'hunan3':
        def preprocess(sample):
            for i in range(13):
                sample["modality1"][i] = (sample["modality1"][i] - torch.min(sample["modality1"][i])) / (
                    torch.max(sample["modality1"][i]) - torch.min(sample["modality1"][i]))

            for i in range(2):
                sample["modality2"][i] = (sample["modality2"][i] - torch.min(sample["modality2"][i])) / (
                    torch.max(sample["modality2"][i]) - torch.min(sample["modality2"][i]))

            max = 1892.0
            min = 18.0
            sample["modality3"] = (sample["modality3"] - min) / (max - min)
            sample["modality3"] = torch.clip(sample["modality3"], min=0.0, max=1.0)

            return sample

        transforms = T.Compose([preprocess])
        train_dataset = Hunan3(root=opts.data_root, split="train", transforms=transforms)
        val_dataset = Hunan3(root=opts.data_root, split="val", transforms=transforms)

    else:
        raise RuntimeError("Dataset not found")
    return train_dataset, val_dataset


def validate(opts, model, criterion, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    if opts.save_val_results:
        if not os.path.exists('results'):
            os.mkdir('results')

        img_id = 0

    with torch.no_grad():
        for i, sample in tqdm(enumerate(loader)):

            modality1 = sample['modality1'].to(device, dtype=torch.float32)
            modality2 = sample['modality2'].to(device, dtype=torch.float32)
            modality3 = sample['modality3'].to(device, dtype=torch.float32)

            labels = sample['mask'].to(device, dtype=torch.long)

            outputs = model(modality1, modality2, modality3)
            loss = criterion(outputs, labels)
            loss = loss.cpu().numpy()
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (modality1[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results and i % 500 == 0:
                for i in range(len(modality1)):
                    image = modality1[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (image * 255).transpose(1, 2, 0).astype(np.uint8)
                    target = target.astype(np.uint8)
                    pred = pred.astype(np.uint8)

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
    return score, ret_samples, loss


def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'hunan' or opts.dataset.lower() == 'hunan2':
        opts.num_classes = 7
    elif opts.dataset.lower() == 'potsdam':
        opts.num_classes = 6
    elif opts.dataset.lower() == 'dfc20':
        opts.num_classes = 11
    elif opts.dataset.lower() == 'passau':
        # TODO: num_classes = 2? adjust for regression instead?
        opts.num_classes = 2
    elif opts.dataset.lower() == 'hunan3':
        opts.num_classes = 7
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

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=4,
                                   drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=4, drop_last=True)

    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](dataset=opts.dataset, num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)

    utils.set_bn_momentum(model.backbone1, momentum=0.01)
    utils.set_bn_momentum(model.backbone2, momentum=0.01)
    utils.set_bn_momentum(model.backbone3, momentum=0.01)
    # Set up optimizer
    optimizer = torch.optim.SGD(params=[
        {'params': model.backbone1.parameters(), 'lr': opts.lr},
        {'params': model.backbone2.parameters(), 'lr': opts.lr},
        {'params': model.backbone3.parameters(), 'lr': opts.lr},
        {'params': model.classifier.parameters(), 'lr': opts.lr},
    ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization

    with open("val_loss/%s_%s.txt" % (opts.model, opts.dataset), 'w', encoding='utf-8') as f:
        f.writelines("\n")

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        cur_epochs += 1
        for sample in train_loader:
            cur_itrs += 1
            modality1 = sample['modality1'].to(device, dtype=torch.float32)
            modality2 = sample['modality2'].to(device, dtype=torch.float32)
            modality3 = sample['modality3'].to(device, dtype=torch.float32)
            if opts.dataset == "gid":
                modality2 = rearrange(modality2, "b h w -> b () h w")
            labels = sample['mask'].to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(modality1, modality2, modality3)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)
            if (cur_itrs) % opts.val_interval == 0:
                interval_loss = interval_loss / opts.val_interval
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))

                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples, loss = validate(
                    opts=opts, model=model, criterion=criterion, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                with open("val_loss/%s_%s.txt" % (opts.model, opts.dataset,), 'a', encoding='utf-8') as f:
                    f.writelines('cir_itrs:' + str(cur_itrs) + metrics.to_str(val_score) + '\n' + 'train_loss:' + str(interval_loss) + '\n' + 'val_loss:' + str(loss) + '\n')
                interval_loss = 0.0
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']

                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))

                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                model.train()
            scheduler.step()
            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()
