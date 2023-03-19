from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

from models.backbone import *
import dataset.affwild as dataset
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='SSL-DTM Training')
# Optimization options
parser.add_argument('--model-name', default='resnet18', type=str, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--image-size', default = 224, type=int,metavar='N',
                    help = 'image size')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch-size', default=256, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
parser.add_argument('--sc', action='store_true')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=5, help='manual seed')
# Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
# Method options
parser.add_argument('--train-iteration', type=int, default=800,
                        help='Number of iteration per epoch')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--ema-decay', default=0.999, type=float)
# Data
parser.add_argument('--train-csv', type=str, default='data/train.csv',
                        help="root path to train data directory")
parser.add_argument('--unlabel-csv', type=str, default='data/unlabeled.csv',
                        help="root path to unlabeled data directory")                       
parser.add_argument('--val-csv', type=str, default='data/val.csv',
                        help="root path to test data directory")


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
best_f1 = 0 # best f1 score
def main():
    global best_acc
    global best_f1

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing Dataset')
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform_train = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.RandomApply([
            transforms.RandomCrop(args.image_size, padding=8)
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_labeled_set, train_unlabeled_set, test_set = dataset.get_affwild(args.train_csv, args.val_csv, args.unlabel_csv, transform_train=transform_train, transform_val=transform_val)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
    print("==> Creating Model")

    def create_model(ema=False,model_name = 'resnet18'):
        if model_name == 'efficientnetb7':
            print('==> Using efficientnetb7')
            model = Efficientb7_timm()
        model = torch.nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()
        return model
    
    model = create_model(model_name=args.model_name)
    ema_model = create_model(ema=True,model_name=args.model_name)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    print(args.out)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay)
    schedu = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                milestones=[10, 15, 20, 25], gamma=0.2)
   
    logger = Logger(os.path.join(args.out, 'log.txt'), title='RAF')
    logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Train Loss S', 'Test Loss', 'Test Acc.', 'F1 Score'])

    test_accs = []
    threshold = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    # threshold = [0.8, 0.6, 0.6, 0.6, 0.8, 0.8, 0.6, 0.8]
    start_epoch = 1
    if args.resume:
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict['state_dict'])
        ema_model.load_state_dict(state_dict['ema_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch']
        # threshold = [0.8592, 0.9015, 0.9088, 0.9073, 0.9273, 0.9161, 0.8968, 0.9563]
    # Train and val
    for epoch in range(start_epoch, args.epochs + 1):

        print('\nEpoch: [%d | %d] LR: %f Threshold=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]' % (epoch, args.epochs, optimizer.state_dict()['param_groups'][0]['lr'], threshold[0], threshold[1], threshold[2], threshold[3], threshold[4], threshold[5], threshold[6], threshold[7]))

        train_loss, train_loss_x, train_loss_u, train_loss_sim = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, threshold, epoch, use_cuda)
        _, train_acc, outputs_new, targets_new, f1score = validate(labeled_trainloader, ema_model, criterion, use_cuda, mode='Train Stats')
        threshold = dym(outputs_new, targets_new, threshold, epoch)

        test_loss, test_acc, _, _, f1score = validate(test_loader, ema_model, criterion, use_cuda, mode='Test Stats')

        if args.sc:
            schedu.step()
        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, train_loss_sim, test_loss, test_acc, f1score])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        test_accs.append(test_acc)

        # save best f1 score model
        if f1score > best_f1:
            best_f1 = f1score
            torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, args.out+'/best_f1_'+'checkpoint.pth.tar')
    logger.close()

    print('Best acc:')
    print(best_acc)


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion_ce, threshold, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2, inputs_strong), _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_strong), _ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)


        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()
            inputs_strong = inputs_strong.cuda()


        # compute guessed labels of unlabeled samples
        outputs_u = model(inputs_u)
        outputs_u2 = model(inputs_u2)
        p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
        max_probs, max_idx = torch.max(p, dim=1)
        max_idx = max_idx.detach()

        output_x  = model(inputs_x)

        mask = mask_generate(max_probs, max_idx, batch_size, threshold)
        # targets_x -= 1
        targets_x = targets_x.type(torch.int64)
        Lx = criterion_ce(output_x, targets_x).mean()


        output_strong = model(inputs_strong)
        Lu = criterion_ce(output_strong, max_idx) * mask
        Lu = Lu.mean()
        loss = Lx  + Lu * 0.5

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.train_iteration,
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    )
        bar.next()

    bar.finish()
    
    return (losses.avg, losses_x.avg, losses_u.avg)

def validate(valloader, model, criterion, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    outputs_new = torch.ones(1, 8).cuda()
    targets_new = torch.ones(1).long().cuda()

    from sklearn.metrics import f1_score
    preds = []
    true_target =[]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            # targets -= 1
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs = model(inputs)
            # print(targets)
            targets = targets.type(torch.int64)
            loss = criterion(outputs, targets).mean()

            ##
            outputs_new = torch.cat((outputs_new, outputs), dim=0)
            targets_new = torch.cat((targets_new, targets), dim=0)
            ##

            # measure accuracy and record loss
            prec1 = accuracy(outputs, targets, topk=(1, 5))
            # print(outputs)
            # print(targets)

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            
            # f1 score
            preds.extend(outputs.argmax(1).to('cpu').numpy())
            true_target.extend(targets.to('cpu').numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Total: {total:} | Loss: {loss:.4f} | Accuracy: {top1: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        )
            bar.next()
        bar.finish()
        score = f1_score(true_target, preds, average='macro')
        print(score)
    return (losses.avg, top1.avg, outputs_new, targets_new, score)

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

def mask_generate(max_probs, max_idx, batch, threshold):
    mask_ori = torch.zeros(batch).cuda()
    for i in range(8):
        idx = np.where(max_idx.cpu() == i)[0]
        m = max_probs[idx].ge(threshold[i]).float()
        for k in range(len(idx)):
            mask_ori[idx[k]]+=m[k]
    return mask_ori.cuda()

def dym(outputs, targets, threshold, epoch):
    outputs_l = outputs[1:, :]
    targets_l = targets[1:]
    probs = torch.softmax(outputs_l, dim=1)
    max_probs, max_idx = torch.max(probs, dim=1)
    eq_idx = np.where(targets_l.eq(max_idx).cpu() == 1)[0]

    probs_new = max_probs[eq_idx]
    targets_new = targets_l[eq_idx]
    for i in range(8):
        idx = np.where(targets_new.cpu() == i)[0]
        if idx.shape[0] != 0:
            threshold[i] = threshold[i] * 0.9 + probs_new[idx].mean().cpu() * 0.1 if probs_new[idx].mean().cpu() >=0.8 else 0.8
        else:
            threshold[i] = 0.8
    return threshold

if __name__ == '__main__':
    main()
