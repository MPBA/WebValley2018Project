import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import sys


# in this code, the word "slice" refers to a single scan taken from a plane of a .nii 3D scan
# that is then paried with it's counterpart taken from the provided labeled .nii
# these pairs are resized to 240, 240 in the event they are smaller by adding padding at the edges
# they are then saved as a 2D tensor with 2 channels, the first being the original brain scan, and 
# the second it's labeled counterpart


CUDA_DEV = 'cuda:0'

device = torch.device(CUDA_DEV if torch.cuda.is_available() else 'cpu')

batch_size = 1


class BRATSDataset(Dataset):
    def __init__(self, section):
        self.section = section
        self.cur_cube = 209        # refers to the current patient, starting as the total number of patients - 1
        self.slices = []
        if section == 'train':
            self.length = 8575     # total number of slices being trained on 
        elif section == 'test':
            self.length = 2947     # total number of slices being tested on
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        while len(self.slices) == 0:      # works by loading the slices from an individual patient into 
            self.fill_list()              # a list, and then passing them to the loader, removing them from 
                                          # the list and refilling the list with scans from the next patient
        output = self.slices[-1]          # when it is empty
        self.slices.pop()
        image = output.narrow(0, 0, 1)    # here the slices are separated into the two parts
        mask = output.narrow(0, 1, 1)
        return image, mask 
            
        
    def fill_list(self):        
        path = f'./data/{self.section}/{self.cur_cube}/flair/'   # the path to the slices of the current patient
        for x in os.listdir(path):
            self.slices.append(torch.load(os.path.join(path, x)))# loads the slices and ands them to the list
            
        self.cur_cube -= 1                                       # changes the current patient
        if self.cur_cube < 0:                                    # resets it when all the patients have
            self.cur_cube = 209                                  # have been analysed
    


class UNet(nn.Module):

    def __init__(self):

        super(UNet, self).__init__()

        
        self.conv1 = nn.Sequential(
                    nn.Conv2d(1, 64, kernel_size = 3, padding = 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
                    nn.ReLU())
        
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv2 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
                    nn.ReLU())
        
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU())
        
        self.max_pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.ReLU())
        
        self.max_pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size = 3, padding = 1),
            nn.ReLU())
        
        self.upconv1 = nn.Upsample(scale_factor = 2)
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            nn.ReLU())
        
        self.upconv2 = nn.Upsample(scale_factor = 2)
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size = 3, padding = 1),
            nn.ReLU())
        
        self.upconv3 = nn.Upsample(scale_factor = 2)
        
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
            nn.ReLU())
        
        self.upconv4 = nn.Upsample(scale_factor = 2)
        
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1),
            nn.ReLU())
        
        self.outconv = nn.Conv2d(64, 1, kernel_size = 1, stride = 1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = self.max_pool2(x)

        x = self.conv3(x)
        x = self.max_pool3(x)

        x = self.conv4(x)
        x = self.max_pool4(x)

        x = self.conv5(x)

        x = self.upconv1(x)
        
        x = self.conv6(x)
        x = self.upconv2(x)

        x = self.conv7(x)
        x = self.upconv3(x)

        x = self.conv8(x)
        x = self.upconv4(x)

        x = self.conv9(x)
        x = self.outconv(x)
        
        return x
        



test_loader = DataLoader(dataset=BRATSDataset('test'),                 # prepares the test loader
                         batch_size = batch_size, shuffle = True)      # with the correct data

print('dataloader ready')

def dice_coefficient(truth, prediction):                               # calculates dice score
    if torch.sum(truth) == 0 and torch.sum(prediction) == 0:           # that goes from 0  to 1
        return 1                                                       # 1 = perfect prediction
    
    if torch.sum(truth) == 0 and torch.sum(prediction) != 0: 
        return 0                                                       # 0 = entirely incorrect prediction
    
    truth = truth.detach().squeeze().cpu().numpy()
    prediction = prediction.detach().squeeze().cpu().numpy()
    
    
    return 2 * np.sum(truth * prediction)/(np.sum(truth) + np.sum(prediction))


MODEL_PATH = './trained on whole brain slices/trained on flair slices/UNet_all_brain_slices_30.pt'

model = torch.load(MODEL_PATH).to(device)


def test():
    
    model.eval()
    criterion = dice_coefficient

    num_steps_test = len(test_loader)
    print(num_steps_test)   
    
#     test_output_file = open('test_dice_score_log.txt', 'w')   # option to print results of each batch to file
    
    num_images = 0
    total_score = 0.0
    
    t0 = time.time()
    print(' - testing - ')
    for i, (images, masks) in enumerate(test_loader):
        num_images += 1
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        outputs += 0.5
        outputs = outputs.int()     # rounds generated image to intergers
        x = torch.zeros(240, 240)
        y = torch.ones(240, 240)
        outputs = torch.where(outputs.detach().squeeze().cpu() > 0, y, x)  # transforms predicted and actual
        masks = torch.where(masks.detach().squeeze().cpu() > 0, y, x)      # mask into binary image
        dice_score = criterion(masks, outputs)
        total_score += dice_score
        sys.stdout.write('\r{:.2f}% - dice score: {:.4f} - ({:.2f}s)'
                     .format(i * 100 / num_steps_test, float(dice_score), time.time() - t0))
        
#         test_output_file.write('Epoch: 0' + str(epoch) + ' Dice score: ' + str(dice_score) + '\n')
        
        if i % 11 == 0:      # condition so as to show only a portion of images
            plt.subplot(1, 4, 1)
            plt.axis('off')
            plt.imshow(images[0].detach().squeeze().cpu().numpy(), cmap = 'gray')

            plt.subplot(1, 4, 2)
            plt.axis('off')
            plt.imshow(outputs.detach().squeeze().cpu().numpy(), cmap = 'gray')

            plt.subplot(1, 4, 3)
            plt.axis('off')
            plt.imshow(masks.detach().squeeze().cpu().numpy(), cmap = 'gray')

            plt.subplot(1, 4, 4)
            plt.axis('off')
            
            x = torch.zeros(240, 240)
            y = torch.ones(240, 240)
            difference = torch.from_numpy(np.subtract(outputs.numpy(), masks.numpy())) # subtracts predicted
                                                                                       # from actual mask
 
            plt.imshow(torch.where(difference != 0, y, x), cmap = 'gray')      # shows all points where the        
                                                                               # prediction was incorrect
            plt.show()
        
        del outputs
        del images
        del masks
    
    print('\naverage dice score over validation images: ' + str(total_score / num_images))
    
    
    
test()   # runs the testing function