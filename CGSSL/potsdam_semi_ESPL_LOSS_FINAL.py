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
import torch.nn.functional as F

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
    parser.add_argument('--exp', type=str, default='potsdam/ESPL_loss_FINAL', help='experiment_name')
#----------------------unet  deeplabv3plus  pspnet   segnet
    parser.add_argument("--model_folder", type=str, default='checkpoints')        
    parser.add_argument("--num_tra_sam", type=str, default='2downsample_train')  
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--total_itrs", type=int, default=38000,
                        help="epoch number (default: 30k)")
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
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    parser.add_argument("--gpu_id", type=str, default='0', help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1, help="random seed (default: 1)")   
    parser.add_argument('--device', type=str,  default='cuda',
                    help='computing device')

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
        db_train = Potsdam2D(root=opts.data_root, num_tra_sam=opts.num_tra_sam, split="train",transforms=transforms)

        val_length = int(len(db_train) * 0.1)
        train_length = len(db_train) - val_length
        db_val, db_train = torch.utils.data.random_split(db_train, [val_length, train_length])

        db_train_unl = Potsdam2D(root=opts.data_root, num_tra_sam=opts.num_tra_sam, split="unlabeled_train",transforms=transforms)
        db_test = Potsdam2D(root=opts.data_root, num_tra_sam=opts.num_tra_sam, split="test", transforms=transforms)
        
    else:
        raise RuntimeError("Dataset not found")
    return db_train, db_train_unl, db_val, db_test

def validate(opts, model1, model2, model3, ce_loss, loader, device, metrics):
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
            loss_ce = ce_loss(outputs,labels)
            supervised_loss = loss_ce.cpu().numpy()
            average_loss = average_loss + supervised_loss

            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            metrics.update(targets, preds)
         
        average_loss = average_loss/number
        score = metrics.get_results()
    return score, average_loss


class ESPL_CE(nn.Module):
    def __init__(self,THR):
        super(ESPL_CE, self).__init__()
        self.THR = THR
    def forward(self, yHat, y, confidence):
        P_i = torch.nn.functional.log_softmax(yHat, dim=1)
        y = torch.nn.functional.one_hot(y,num_classes=6)
        y = y.permute(0, 3, 1, 2)
        loss = y*(P_i + 0.000000000001)
        loss = torch.sum(loss, dim=1)
        mask = confidence > torch.mean(confidence)
        print("how many pixels are masked: "+ str(torch.numel(confidence[mask])))
        confidence = (torch.max(confidence)-confidence) /(torch.max(confidence)-torch.min(confidence))
        confidence = confidence + 1
        loss[mask] = loss[mask] * confidence[mask]  

        loss = torch.mean(loss, dim=(0,1,2))
        hand_cross_entropy = -1*loss
        return hand_cross_entropy

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
    db_train, db_train_unl, db_val, db_test = get_dataset(opts)
    train_label_loader = data.DataLoader(db_train, batch_size=opts.batch_size, shuffle=True, num_workers=0,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    train_unlabel_loader = data.DataLoader(db_train_unl, batch_size=opts.batch_size, shuffle=True, num_workers=0,
        drop_last=True)
    val_loader = data.DataLoader(
        db_val, batch_size=opts.val_batch_size, shuffle=False, num_workers=0, drop_last=True)
    test_loader = data.DataLoader(
        db_test, batch_size=opts.val_batch_size, shuffle=False, num_workers=0, drop_last=True)

    print("Dataset: %s, labeled train set: %d, unlabeled train set: %d, Val set: %d" %
          (opts.dataset, len(db_train), len(db_train_unl), len(db_val)))

    # Set up model (all models are 'constructed at network.modeling)
    # model1 = create_model(opts, ema=False) #unet
    model1 = network.modeling.__dict__['unet'](in_number=opts.in_number, num_classes=opts.num_classes)
    model2 = network.modeling.__dict__['segnet'](in_number=opts.in_number, num_classes=opts.num_classes)
    model3 = network.modeling.__dict__['pspnet'](in_number=opts.in_number, num_classes=opts.num_classes)

    # Set up optimizer
    optimizer1 = torch.optim.SGD(params=model1.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    optimizer2 = torch.optim.SGD(params=model2.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    optimizer3 = torch.optim.SGD(params=model3.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    if opts.lr_policy == 'poly':
        scheduler1 = utils.PolyLR(optimizer1, opts.total_itrs, power=0.9)
        scheduler2 = utils.PolyLR(optimizer2, opts.total_itrs, power=0.9)
        scheduler3 = utils.PolyLR(optimizer3, opts.total_itrs, power=0.9)

    #set loss functions
    if opts.dataset == "potsdam":
        ce_loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean', size_average = True)
        espl_ce_loss = ESPL_CE(0.8)
    else:
        raise RuntimeError("Dataset not found")

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    def save_ckpt1(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model1.module.state_dict(),
            "optimizer_state": optimizer1.state_dict(),
            "scheduler_state": scheduler1.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    def save_ckpt2(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model2.module.state_dict(),
            "optimizer_state": optimizer2.state_dict(),
            "scheduler_state": scheduler2.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
    
    def save_ckpt3(path):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model3.module.state_dict(),
            "optimizer_state": optimizer3.state_dict(),
            "scheduler_state": scheduler3.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        model1 = nn.DataParallel(model1)
        model2 = nn.DataParallel(model2)
        model3 = nn.DataParallel(model3)
        model1.to(device)
        model2.to(device)
        model3.to(device)

    # store information
    snapshot_path = "../model/{}_".format(opts.exp)
    if os.path.exists(snapshot_path) and os.path.isdir(snapshot_path):
        shutil.rmtree(snapshot_path)
        os.makedirs(snapshot_path)
    else:
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))
    os.makedirs(snapshot_path+'/checkpoints')
    writer_train = SummaryWriter(snapshot_path + '/log_train')
    writer_val = SummaryWriter(snapshot_path + '/log_val')
    # ==========   Train Loop   ==========#
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    interval_loss = 0.0
    max_epoch = opts.total_itrs // len(train_label_loader) + 1
    while True:  
        unlabel_dataloader = iter(train_unlabel_loader)
        model1.train()
        model2.train()
        model3.train()
        cur_epochs += 1
        for nimibatch_l in tqdm(train_label_loader):
            try:
                nimibatch_unl = next(unlabel_dataloader)
            except StopIteration:
                unlabel_dataloader = iter(train_unlabel_loader)
                nimibatch_unl = next(unlabel_dataloader)
            
            labeled_image = nimibatch_l['image'].to(device, dtype=torch.float32)
            label = nimibatch_l['label'].to(device, dtype=torch.long)

            cur_itrs += 1

            if cur_itrs < 3500:
                pred_sup_1 = model1(labeled_image)
                pred_sup_2 = model2(labeled_image)
                pred_sup_3 = model3(labeled_image)

                loss_sup_1 = ce_loss(pred_sup_1, label)
                loss_sup_2 = ce_loss(pred_sup_2, label)
                loss_sup_3 = ce_loss(pred_sup_3, label)

                loss = loss_sup_1 + loss_sup_2 + loss_sup_3 

                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()

                
            else:
                unlabeled_image = nimibatch_unl['image'].to(device, dtype=torch.float32)
                pred_sup_1 = model1(labeled_image)
                pred_sup_2 = model2(labeled_image)
                pred_sup_3 = model3(labeled_image)

                pred_unsup_1 = model1(unlabeled_image)
                pred_unsup_2 = model2(unlabeled_image)
                pred_unsup_3 = model3(unlabeled_image)

                pred_1 = torch.cat([pred_sup_1, pred_unsup_1], dim=0)
                pred_2 = torch.cat([pred_sup_2, pred_unsup_2], dim=0)
                pred_3 = torch.cat([pred_sup_3, pred_unsup_3], dim=0)

                max_1 = torch.max(pred_1, dim=1)[1].long()
                max_2 = torch.max(pred_2, dim=1)[1].long()
                max_3 = torch.max(pred_3, dim=1)[1].long()

                pred = pred_1 + pred_2 + pred_3
                pred_softmax = torch.softmax(pred,dim=1)
                
                temp = pred_softmax.permute(0, 2, 3, 1)
                confidence = Categorical(probs = temp).entropy()
                cps_loss = espl_ce_loss(pred_1, max_3, confidence) + espl_ce_loss(pred_1, max_2, confidence) + espl_ce_loss(pred_2, max_1, confidence) + espl_ce_loss(pred_2, max_3, confidence) + espl_ce_loss(pred_3, max_1, confidence) + espl_ce_loss(pred_3, max_2, confidence)
                cps_loss = cps_loss * 1.5 
                loss_sup_1 = ce_loss(pred_sup_1, label)
                loss_sup_2 = ce_loss(pred_sup_2, label)
                loss_sup_3 = ce_loss(pred_sup_3, label)

                loss = loss_sup_1 + loss_sup_2 + loss_sup_3 + cps_loss 
                
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                loss.backward()
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()

                np_cps_loss = cps_loss.detach().cpu().numpy()
                writer_train.add_scalar('info/cps_loss', np_cps_loss.item(), cur_itrs)

            np_loss_sup_1 = loss_sup_1.detach().cpu().numpy()
            np_loss_sup_2 = loss_sup_2.detach().cpu().numpy()
            np_loss_sup_3 = loss_sup_3.detach().cpu().numpy()
            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            writer_train.add_scalar('info/loss_sup_1', np_loss_sup_1.item(), cur_itrs)
            writer_train.add_scalar('info/loss_sup_2', np_loss_sup_2.item(), cur_itrs)
            writer_train.add_scalar('info/loss_sup_3', np_loss_sup_3.item(), cur_itrs)
            writer_train.add_scalar('info/loss_sup_1+2+3', np_loss_sup_1.item() + np_loss_sup_2.item() + np_loss_sup_3.item(), cur_itrs)
            writer_train.add_scalar('info/loss', np_loss.item(), cur_itrs)  
            
            if (cur_itrs) % opts.val_interval == 0:
                interval_loss = interval_loss / opts.val_interval
                writer_train.add_scalar('info/interval_loss', interval_loss.item(), cur_itrs)
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                save_ckpt1(snapshot_path+'/'+opts.model_folder+'/latest_%s_model1.pth' %
                                (opts.dataset))
                save_ckpt2(snapshot_path+'/'+opts.model_folder+'/latest_%s_model2.pth' %
                                (opts.dataset))
                save_ckpt3(snapshot_path+'/'+opts.model_folder+'/latest_%s_model3.pth' %
                                (opts.dataset))
                print("validation...")
                model1.eval()
                model2.eval()
                model3.eval()
                val_score, loss = validate(
                    opts=opts, model1=model1, model2=model2, model3=model3, ce_loss=ce_loss,  loader=val_loader, device=device, metrics=metrics)
                print(metrics.to_str(val_score))
                writer_val.add_scalar('info/loss', loss.item(), cur_itrs)
                writer_val.add_scalar('info/interval_loss', loss.item(), cur_itrs)
                writer_val.add_scalar('info/mIoU',val_score['Mean IoU'], cur_itrs)
                writer_val.add_scalar('info/Overall_Acc', val_score['Overall Acc'], cur_itrs)
                writer_val.add_scalar('info/UA', val_score['UA'], cur_itrs)
                writer_val.add_scalar('info/PA', val_score['PA'], cur_itrs)
                writer_val.add_scalar('info/f1', val_score['f1'], cur_itrs)

                interval_loss = 0.0
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt1(snapshot_path+'/'+opts.model_folder+'/best_%s_%d_model1.pth' %
                              (opts.dataset, cur_itrs))
                    save_ckpt2(snapshot_path+'/'+opts.model_folder+'/best_%s_%d_model2.pth' %
                              (opts.dataset, cur_itrs))
                    save_ckpt3(snapshot_path+'/'+opts.model_folder+'/best_%s_%d_model3.pth' %
                              (opts.dataset, cur_itrs))
                model1.train()
                model2.train()
                model3.train()
            scheduler1.step()
            scheduler2.step()
            scheduler3.step()
            if cur_itrs >= opts.total_itrs:
                writer_train.close()
                writer_val.close()
                return
if __name__ == '__main__':
    main()
