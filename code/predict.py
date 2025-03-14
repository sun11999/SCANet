import argparse
import os
import shutil
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data as Data
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
from models import *
import real_data
# import get_spectrum_data
import pandas as pd
from sklearn import metrics
import pickle


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser(description='PyTorch SCANet predicting')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--resume', default='model_best.pth.tar', type=str,
                    help='path to latest checkpoint (default: none):  model_best.pth.tar  checkpoint.pth')
parser.add_argument('--name', default='SCANet', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--data', default='Mn', type=str,
                    help='name of data: Mn, Mn_dx, Mn_dl')                    

best_acc = 0
best_mcc = 0
best_auroc = 0
best_sen = 0
best_spe = 0

def main():
    global args, best_prec1
    args = parser.parse_args()
    # if args.tensorboard: configure("runs/%s"%(args.name))
    
    # Data loading code
    kwargs = {'num_workers': 8, 'pin_memory': True}       
    #datasets
    test_dataset = real_data.FeatureDataset(args, mode = 'test',type=args.data)
    test_loader = real_data.DataLoaderX(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
         

    # create model
    if args.data == 'Mn':
        cls = 2
    elif args.data == 'Mn_dx':
        cls = 3
    elif args.data == 'Mn_dl':    
        cls = 1
    else:
        print("unrecognized task!")
    
    model = SCANet(BasicBlock, [2, 2, 2, 2], num_classes=cls)
    
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    model = model.cuda()

    # optionally resume from a checkpoint
    dir = "runs/%s_%s"%(args.name,args.data)
    save_dir = os.path.join(dir,args.resume)

    if args.resume:
        if os.path.isfile(save_dir):
            checkpoint = torch.load(save_dir)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' "
                  .format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    criterion = nn.BCEWithLogitsLoss().cuda()

    # evaluate on test set
    test(test_loader, model, criterion)



def test(test_loader, model, criterion):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    incorrect = 0
    Corr = 0
    auroc = 0
    total_correct = 0
    total_num = 0    


    # switch to evaluate mode
    model.eval()
    
    y_true = []
    y_pred = []
    lens_list = []
    lens_right = []
    lens_wrong = []
    l_r =0
    l_w =0
    a=0

    end = time.time()
    for i, (target) in enumerate(test_loader):        

        target = target.type(torch.FloatTensor)                 
        target = target.cuda() 
      
        with torch.no_grad():
            target_var = torch.autograd.Variable(target)

        # spectra metrics learning
        all = np.load('features/features_Mn.npz')
        all_label = np.load('features/labels_Mn.npz')

        all = torch.from_numpy(all['arr_0'] )
        all = all.type(torch.FloatTensor)
        all_label = torch.from_numpy(all_label['arr_0'] )
        all_label = all_label.type(torch.FloatTensor)        
        all = all.cuda()
        all_label = all_label.cuda()
        
        # compute output
        output = model.predict(target_var,all, all_label)
        
        # measure metrics       
        output1 = output.cpu()
        #reg
        # output1 = np.maximum(output1.detach().numpy(),0)
        #class
        # output2 = np.argmax(output1.detach().numpy(), axis=1)
        
        # #save class probability
        # df1 = pd.DataFrame(output1.detach().numpy())
        # df1.to_excel('class_pro.xlsx',  
            # index=False,         
            # engine='openpyxl')   

        y_pred.extend(output1.tolist())



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
 
    
if __name__ == '__main__':
    main()
