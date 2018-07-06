import torch
import torch.nn as nn
import SimpleITK as sitk
import numpy as np
import os
import sys

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# in this code, the word "slice" refers to a single scan taken from a plane of a .nii 3D scan
# that is then paried with it's counterpart taken from the provided labeled .nii
# these pairs are resized to 240, 240 in the event they are smaller by adding padding at the edges
# they are then saved as a 2D tensor with 2 channels, the first being the original brain scan, and 
# the second it's labeled counterpart

class BRATSDataset(Dataset):
    def __init__(self, dirs):
        self.cur_cube = len(dirs) - 1   # refers to the current patient, initialized to the last
        self.save_dirs = dirs
        self.dirs = dirs
        self.flair_assiale = []         # the three lists that will contain the various slices 
        self.flair_coronale = []        # taken from the different planes of the 3D scans
        self.flair_sagitale = []        # that will then get saved to file
        
        self.length = len(dirs)         # total number of patients
        
    def __getitem__(self, index):
        
        self.get_cube()                 # retrieves the next patient, emptying the lists from the previous
        while len(self.flair_assiale):
            self.flair_assiale.pop()
        while len(self.flair_coronale):
            self.flair_coronale.pop()
        while len(self.flair_sagitale):
            self.flair_sagitale.pop()
            
        return 1, 1                     # returns a non important value
    
        
    def __len__(self):
        return self.length
    
    def get_filename(self, path, casename, img_type):
        return os.path.join(path, '%s_%s.nii.gz' % (casename, img_type))
    
    def get_cube(self):   # loads a new patient
        path = self.dirs[self.cur_cube]
        self.cur_cube -= 1                          #prepares for the next patient, resetting if needed
        if self.cur_cube < 0:
            self.cur_cube = len(self.dirs) - 1
            
        SCAN_TYPE = 'flair'   # types of scan in BraTS: 't1', 't1ce', 't2', 'flair'
        
        # retrieves the  MRI scan and the labeled data and converts both into tensors
        
        casename = os.path.basename(os.path.normpath(path))
        flair_filenames = [self.get_filename(path, casename, SCAN_TYPE)]  
        seg_filename = [self.get_filename(path, casename, 'seg')]         
        imgs = [sitk.ReadImage(el) for el in flair_filenames]            
        arrs = [sitk.GetArrayFromImage(el) for el in imgs]
        tensors = [torch.from_numpy(el) for el in arrs]
        tensors = [el.unsqueeze(0) for el in tensors]
        images = torch.cat(tensors, 0).to(dtype=torch.float32)
        masks = sitk.ReadImage(seg_filename[0])
        masks = torch.from_numpy(sitk.GetArrayFromImage(masks))
        
        images = images.squeeze()
        masks = masks.squeeze()
        
        self.slicer(images, masks)
        self.separate_slices()
        
    def get_dir(self):
        return self.dirs
    
    def slicer(self, image, masks):   # retrieves the slices of scan that contain more that a set amount of brain
                                      # these threshold values were defined so as to have a roughly equal number of
                                      # slices from each plane
        #axial
        for iii in range(image.size(0)):
            segment = image.narrow(0, iii, 1)
            mask = masks.narrow(0, iii, 1)
            segment = segment.squeeze()
            segment = segment.unsqueeze(0)
            mask = mask.type(torch.FloatTensor)  # mask is converted to the same data type as the image
            output = torch.cat((segment, mask))
            if(segment.sum() >5.1e6):      
                self.flair_assiale.append(output)

        #coronal
        for iii in range(image.size(1)):
            segment = image.narrow(1, iii, 1)
            mask = masks.narrow(1, iii, 1)
            segment = segment.squeeze()
            segment = segment.unsqueeze(0)
            mask = mask.squeeze()
            mask = mask.unsqueeze(0)
            mask = mask.type(torch.FloatTensor) # mask is converted to the same data type as the image
            output = torch.cat((segment, mask))
            output = torch.cat((output, torch.zeros(2, 85, 240)), 1) # the tensor is resized to 240 by 240
            if(segment.sum() > 4.4e6):
                self.flair_coronale.append(output)
        #sagittal
        for iii in range(image.size(2)):
            segment = image.narrow(2, iii, 1)
            mask = masks.narrow(2, iii, 1)
            segment = segment.squeeze()
            segment = segment.unsqueeze(0)
            mask = mask.squeeze()
            mask = mask.unsqueeze(0)
            mask = mask.type(torch.FloatTensor) # mask is converted to the same data type as the image
            output = torch.cat((segment, mask))
            output = torch.cat((output, torch.zeros(2, 85, 240)), 1) # the tensor is resized to 240 by 240
            if(segment.sum() > 4.5e6):
                self.flair_sagitale.append(output)
    
    
    def separate_slices(self):
        
        
        if not os.path.isdir(f'./data/train/{self.cur_cube}/'):           # creates a file for each patient
            os.mkdir(f'./data/train/{self.cur_cube}/')                    # that contains all the slices saved
        if not os.path.isdir(f'./data/train/{self.cur_cube}/flair/'):     # in a file named after the type of scan
            os.mkdir(f'./data/train/{self.cur_cube}/flair/')
        if not os.path.isdir(f'./data/test/{self.cur_cube}/'):
            os.mkdir(f'./data/test/{self.cur_cube}/')
        if not os.path.isdir(f'./data/test/{self.cur_cube}/flair/'):
            os.mkdir(f'./data/test/{self.cur_cube}/flair/')
            
        train, test = train_test_split(self.flair_assiale, test_size = 0.25)  # splits the data between train and test
        for i, el in enumerate(test):
            torch.save(el, f'./data/test/{self.cur_cube}/flair/assiale_{str(i).zfill(3)}.pt')
            #print('./data/test/{self.cur_cube}/flair/assiale_{str(i).zfill(3)}.pt')
            
        for i, el in enumerate(train):
            torch.save(el, f'./data/train/{self.cur_cube}/flair/assiale_{str(i).zfill(3)}.pt')
            #print('./data/train/{self.cur_cube}/flair/assiale_{str(i).zfill(3)}.pt')
            
        train, test = train_test_split(self.flair_coronale, test_size = 0.25)  # splits the data between train and test
        for i, el in enumerate(test):
            torch.save(el, f'./data/test/{self.cur_cube}/flair/coronale_{str(i).zfill(3)}.pt')
            #print('./data/test/{self.cur_cube}/flair/coronale_{str(i).zfill(3)}.pt')
            
        for i, el in enumerate(train):
            torch.save(el, f'./data/train/{self.cur_cube}/flair/coronale_{str(i).zfill(3)}.pt')
            #print('./data/train/{self.cur_cube}/flair/coronale_{str(i).zfill(3)}.pt')
            
        train, test = train_test_split(self.flair_sagitale, test_size = 0.25)  # splits the data between train and test
        for i, el in enumerate(test):
            torch.save(el, f'./data/test/{self.cur_cube}/flair/sagitale_{str(i).zfill(3)}.pt')
            #print('./data/test/{self.cur_cube}/flair/sagitale_{str(i).zfill(3)}.pt')
            
        for i, el in enumerate(train):
            torch.save(el, f'./data/train/{self.cur_cube}/flair/sagitale_{str(i).zfill(3)}.pt')
            #print('./data/train/{self.cur_cube}/flair/sagitale_{str(i).zfill(3)}.pt')
        

        
    
                
                
                

def find_folders(folder):
    subfolders = list(map(lambda x: '/'.join(x), list(filter(lambda x: not x[4].startswith('.'), list(map(
        lambda x: x.split('/'), [f.path for f in os.scandir(folder) if f.is_dir()]))))))
    
    return subfolders    # returns a list containing the file path to each patient


BRATS_PATH = './brats/HGG/'  # path to where the patients are saved

data_loader = DataLoader(dataset=BRATSDataset(find_folders(BRATS_PATH)), 
                          batch_size = 1, shuffle = True)



print('dataloader ready')




for i, (a, b) in enumerate(data_loader):
    a = b  # a simple operation that has no relevance

    
print('storage done')