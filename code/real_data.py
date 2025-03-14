import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms


from PIL import Image, ImageOps, ImageFilter
import os.path as osp
import sys
import random

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
import numpy as np
import pandas as pd
import re

import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from scipy import signal
from scipy import interpolate

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import preprocessing

def SG(data, w=11, p=2):
    """
       :param data: raw spectrum data, shape (n_samples, n_features)
       :param w: int
       :param p: int
       :return: data after SG :(n_samples, n_features)
    """
    return signal.savgol_filter(data, w, p)
    
def nor(data):
    return (data-min(data))/(max(data)-min(data))
    
def jxjz(x,y_uniform):
    n = 5
    p0 = np.polyfit(x,y_uniform,n)

    y_fit0 = np.polyval(p0,x) 

    r0 = y_uniform-y_fit0
    dev0 = np.sqrt(np.sum((r0-np.mean(r0))**2)/len(r0))
    y_remove0 = y_uniform[y_uniform <= y_fit0] 
    x_remove0 = x[np.where(y_uniform <= y_fit0)] 
    
    i=0
    judge=1
    dev=[]
    while judge:
        p1 = np.polyfit(x_remove0, y_remove0, n)  
        y_fit1 = np.polyval(p1, x_remove0)  
        r1 = y_remove0 - y_fit1
        dev1 = np.sqrt(np.sum((r1 - np.mean(r1)) ** 2) / len(r1)) 
        dev.append(dev1)
        if i == 0:
            judge = abs(dev[i] - dev0) / dev[i] > 0.05;
        else:
            judge = abs((dev[i] - dev[i-1]) / dev[i]) > 0.05; 
        y_remove0[np.where(y_remove0 >= y_fit1)] = y_fit1[np.where(y_remove0 >= y_fit1)];
        i=i+1
    y_baseline=np.polyval(p1, x)  
    y_baseline_correction=y_uniform-y_baseline 
    
    return y_baseline_correction

class DataLoaderX(DataLoader):
    def _iter_(self):
        return BackgroundGenerator(super()._iter_())
      
def Load_intensity_3c():
    print('Data Loading...')        
    
    data_dir = 'Mars.xlsx'
    spectra_num = 3
    Myinput = np.zeros([spectra_num, 3, 305], dtype=float)
    
    data = pd.read_excel(data_dir,sheet_name = 0,  engine='openpyxl')    
    
    # interpolate spectrum
    for i in range(spectra_num):        
        wavelengths = data.values[:,0]
        intensity = data.values[:,(i+1)]
        #nor+baseline
        intensity1 = jxjz(wavelengths,nor(SG(intensity)))
        f2 = interpolate.interp1d(wavelengths,intensity1,kind='cubic')
        x_pred = np.linspace(0.865,2.385,num=305)
        y_pred = f2(x_pred)
        
        #original
        Myinput[i][0][:] = y_pred       
        #1st
        y_pred = np.diff(y_pred)         
        Myinput[i][1][1:] = y_pred
        #2st
        y_pred = np.diff(y_pred)  
        Myinput[i][1][1:-1] = y_pred
                                
    SpectrumData = Myinput
    
    return SpectrumData    

class FeatureDataset(Data.Dataset):
    def __init__(self, args, mode='train',type='Mn'):
            
        # get Data
        Myinput = Load_intensity_3c()                                                  
        # print(Myinput.shape)        
        self.data_code = Myinput
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, i):

        encode = self.data_code[i]

        return encode 

    def __len__(self):
        return len(self.data_code)
    

