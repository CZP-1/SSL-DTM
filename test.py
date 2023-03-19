import argparse
import os
import time
import pandas as pd

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

from models.backbone import *
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

def create_model(ema=False,model_name = 'resnet18'):
    if model_name == 'efficientnetb7':
        print('==> Using efficientnetb7')
        model = Efficientb7_timm()


    model = torch.nn.DataParallel(model).cuda()

    if ema:
        for param in model.parameters():
            param.detach_()
    return model

parser = argparse.ArgumentParser(description='Semi-Supervised')
parser.add_argument('--model-name', default='efficientnetb7', type=str, metavar='N',
                    help='number of total epochs to run')
# parser.add_argument('--batch-size', default=64, type=int, metavar='N',
#                     help='train batchsize')
parser.add_argument('--test-batch-size', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--image-size', default = 224, type=int,metavar='N',
                    help = 'image size')
parser.add_argument('--weights', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpu', default='4,5,6', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--out', default='',
                        help='Directory to output the result')
parser.add_argument('--val-csv', type=str, default='',
                        help="root path to test data directory")
parser.add_argument('--mode', type=str, default='val',
                        help="test or val")

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# data
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

if not os.path.isdir(args.out):
        mkdir_p(args.out)

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

from dataset.affwild import Dataset_Affwild
test_dataset = Dataset_Affwild(args.val_csv,transform=transform_val)
test_loader = data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=16)

print("==> Creating Model")

model = create_model(model_name=args.model_name)
state_dict = torch.load(args.weights)

model.load_state_dict(state_dict['ema_state_dict'])
cudnn.benchmark = True
print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print(f"Results will be saved in {args.out}")


batch_time = AverageMeter()
data_time = AverageMeter()

# switch to evaluate mode
model.eval()

end = time.time()
bar = Bar(f'testing', max=len(test_loader))

outputs_new = torch.ones(1, 8).cuda()
outputs_new_softmax = torch.ones(1, 8).cuda()
targets_new = torch.ones(1).long().cuda()

from sklearn.metrics import f1_score,classification_report,confusion_matrix
preds = []
true_target =[]
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
        # compute output
        outputs = model(inputs)
        # print(targets)
        targets = targets.type(torch.int64)

        output_softmax = torch.softmax(outputs,dim=1)

        ##
        outputs_new = torch.cat((outputs_new, outputs), dim=0)
        outputs_new_softmax = torch.cat((outputs_new_softmax, output_softmax), dim=0)
        targets_new = torch.cat((targets_new, targets), dim=0)
        ##
        
        # f1 score
        preds.extend(outputs.argmax(1).to('cpu').numpy())
        true_target.extend(targets.to('cpu').numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} |'.format(
                    batch=batch_idx + 1,
                    size=len(test_loader),
                    total=bar.elapsed_td,
                    )
        bar.next()
    bar.finish()

    if args.mode == 'val':
        report = classification_report(preds, true_target, digits=4)
        print(report)
        df_R = pd.DataFrame([report])
        df_R.to_csv(os.path.join(args.out,'classification_report.csv'))
        C = confusion_matrix(preds, true_target)
        df_C = pd.DataFrame(C)
        df_C.to_csv(os.path.join(args.out,'confusion_matrix.csv'))
        score = f1_score(true_target, preds, average='macro')
        # with open os.path.join(args.out,'f1')

    data_test = pd.read_csv(args.val_csv)
    Id = data_test['path']
    print(len(Id))
    print(len(outputs_new_softmax))
    outputs_new_softmax = outputs_new_softmax[1:, :]
    print(len(outputs_new_softmax))

    print(len(outputs_new))
    outputs_new = outputs_new[1:, :]
    print(len(outputs_new))

    result = pd.DataFrame(preds)
    submission = pd.concat([Id, result], axis=1)
    submission.columns = ['path', 'label']
    result_path = os.path.join(args.out, 'preds.csv')
    submission.to_csv(result_path,index=False, header=True)

    col = [str(i) for i in range(8)] 
    result_soft=np.squeeze(outputs_new.cpu().numpy())
    result_pd = pd.DataFrame(result_soft,columns=col)
    submission_logits = pd.concat([Id, result_pd], axis=1)
    result_logits_path = os.path.join(args.out,'logits.csv')
    submission_logits.to_csv(result_logits_path, index=False, header=True)

    result_softmax=np.squeeze(outputs_new_softmax.cpu().numpy())
    result_pd_softmax = pd.DataFrame(result_softmax,columns=col)
    submission_soft = pd.concat([Id, result_pd_softmax], axis=1)
    result_soft_path = os.path.join(args.out,'softmax.csv')
    submission_soft.to_csv(result_soft_path, index=False, header=True)

