import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
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

epochs = 30
learning_rate = 0.00001
num_classes = 4
batch_size = 16

class BRATSDataset(Dataset):
    def __init__(self, section):
        self.section = section
        self.cur_cube = 209      # last patient
        self.slices = []         # list that will be filled with slices
        if section == 'train':
            self.length = 8575   # number of slices in the training set
        elif section == 'test':
            self.length = 2947   # number of slices in the testing set
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):        
        while len(self.slices) == 0:  # if the list is empty, fill it with new slices
            self.fill_list()
        
        output = self.slices[-1]      # take the last slice in the list, and then remove it from the list
        self.slices.pop()
        image = output.narrow(0, 0, 1)
        mask = output.narrow(0, 1, 1)
        return image, mask            # separate and return the image and segmentation mask
            
        
    def fill_list(self):        
        path = f'./data/{self.section}/{self.cur_cube}/flair/'    # path to the slices of the current patient
        #print(len(os.listdir(path)))
        for x in os.listdir(path):
            self.slices.append(torch.load(os.path.join(path, x))) # load each slice and add it to the list
            
        self.cur_cube -= 1       # prepare for the next person, resetting if necessary
        if self.cur_cube < 0:
            self.cur_cube = 209
    
    
train_loader = DataLoader(dataset=BRATSDataset('train'), 
                          batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset=BRATSDataset('test'), 
                         batch_size = batch_size, shuffle = True)

print('dataloader ready')


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
#         print('start ' + str(x.size()))
        x = self.conv1(x)
#         print('after conv1 ' + str(x.size()))
        x = self.max_pool1(x)
#         print('after max pool 1 ' + str(x.size()))
        x = self.conv2(x)
#         print('after conv2 ' + str(x.size()))
        x = self.max_pool2(x)
#         print('after max pool 2 ' + str(x.size()))
        x = self.conv3(x)
#         print('fter conv 3 ' + str(x.size()))
        x = self.max_pool3(x)
#         print('after max pool 3 ' + str(x.size()))
        x = self.conv4(x)
#         print('after conv 4 ' + str(x.size()))
        x = self.max_pool4(x)
#         print('after max pool 4 ' + str(x.size()))
        x = self.conv5(x)
#         print('after conv 5 ' + str(x.size()))
        x = self.upconv1(x)
#         print('after up conv 1 ' + str(x.size()))
        x = self.conv6(x)
#         print('after conv 6 ' + str(x.size()))
        x = self.upconv2(x)
#         print('after up conv 2 ' + str(x.size()))
        x = self.conv7(x)
#         print('after coonv 7 ' + str(x.size()))
        x = self.upconv3(x)
#         print('after up conv 3 ' + str(x.size()))
        x = self.conv8(x)
#         print('after conv 8 ' + str(x.size()))
        x = self.upconv4(x)
#         print('after up conv 4 ' + str(x.size()))
        x = self.conv9(x)
#         print('after conv 9 ' + str(x.size()))
        x = self.outconv(x)
#         print('after out conv ' + str(x.size()))
        
        return x
    
    
def dice_loss(output, mask):
    probs = output[:, 0, :, :]
    mask = torch.squeeze(mask, 1)

    num = probs * mask
    num = torch.sum(num, 2)
    num = torch.sum(num, 1)

    # print('num : ', num )

    den1 = probs * probs
    # print('den1 : ', den1.size())
    den1 = torch.sum(den1, 2)
    den1 = torch.sum(den1, 1)

    # print('den1 2 : ', den1.size())

    den2 = mask * mask
    # print('den2 : ', den2.size())
    den2 = torch.sum(den2, 2)
    den2 = torch.sum(den2, 1)

    # print('den2 2 : ', den2.size())
    eps = 0.0000001
    dice = 2 * ((num + eps) / (den1 + den2 + eps))
    # dice_eso = dice[:, 1:]
    dice_eso = dice

    loss = 1 - torch.sum(dice_eso) / dice_eso.size(0)
    return loss

MODEL_PATH = '/data/group2/only trained on cancer/UNet from cubes flair/medical_paper_UNet_20.pt'

#model = UNet().to(device)
model = torch.load(MODEL_PATH).to(device)



def train():

    criterion = dice_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_steps_train = len(train_loader)
    num_steps_test = len(test_loader)
    
    print(num_steps_train)
    print(num_steps_test)
    
    TEXT_FILE_SAVE_PATH = './trained on whole brain slices/trained on flair slices'
    
    MODEL_WEIGHTS_SAVE_PATH = './trained on whole brain slices/trained on flair slices/UNet_all_brain_slices_'
    
    train_output_file = open(f'{TEXT_FILE_SAVE_PATH}/train_loss_log_from_file.txt', 'w')
    test_output_file = open(f'{TEXT_FILE_SAVE_PATH}/test_loss_log_from_file.txt', 'w')
    
    for epoch in range(epochs):
        t0 = time.time()
        print(' - training - ')
        model.train()
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sys.stdout.write('\r Epoch {} of {}   {:.2f}% - loss: {:.4f} - ({:.2f}s)'
                       .format(epoch + 1, epochs, i * 100 / num_steps_train, float(loss), time.time() - t0))
            
            # fix the formatting in the text file, so as to ensure that all numbers are in the same column
            
            if 0.095 < loss < 0.1:
                loss = 'tensor(0.1000'
            elif 0.085 < loss < 0.095:
                loss = 'tensor(0.0900'
            elif 0.075 < loss < 0.085:
                loss = 'tensor(0.0800'
            elif 0.065 < loss < 0.075:
                loss = 'tensor(0.0700'
            elif 0.055 < loss < 0.065:
                loss = 'tensor(0.0600'
            elif 0.045 < loss < 0.055:
                loss = 'tensor(0.0500'
            elif 0.035 < loss < 0.045:
                loss = 'tensor(0.0400'
            elif 0.025 < loss < 0.035:
                loss = 'tensor(0.0300'
            elif 0.015 < loss < 0.025:
                loss = 'tensor(0.0200'
            elif loss < 0.015:
                loss = 'tensor(0.0100'
                
            if(epoch < 10):
                train_output_file.write('Epoch: 0' + str(epoch) + ' Loss: ' + str(loss) + '\n')
            
            else:
                train_output_file.write('Epoch: ' + str(epoch) + ' Loss: ' + str(loss) + '\n')
              
            del images
            del masks
            del outputs
            
        print(' - testing - ')
        with torch.no_grad():
            model.eval()
            for i, (images, masks) in enumerate(test_loader):
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)



                sys.stdout.write('\r Testing Epoch {} of {}   {:.2f}% - loss: {:.4f} - ({:.2f}s)'
                             .format(epoch + 1, epochs, i * 100 / num_steps_test, float(loss), time.time() - t0))

                # fix the formatting in the text file, so as to ensure that all numbers are in the same column
                
                if 0.095 < loss < 0.1:
                    loss = 'tensor(0.1000'
                elif 0.085 < loss < 0.095:
                    loss = 'tensor(0.0900'
                elif 0.075 < loss < 0.085:
                    loss = 'tensor(0.0800'
                elif 0.065 < loss < 0.075:
                    loss = 'tensor(0.0700'
                elif 0.055 < loss < 0.065:
                    loss = 'tensor(0.0600'
                elif 0.045 < loss < 0.055:
                    loss = 'tensor(0.0500'
                elif 0.035 < loss < 0.045:
                    loss = 'tensor(0.0400'
                elif 0.025 < loss < 0.035:
                    loss = 'tensor(0.0300'
                elif 0.015 < loss < 0.025:
                    loss = 'tensor(0.0200'
                elif loss < 0.015:
                    loss = 'tensor(0.0100'

                if(epoch < 10):
                    test_output_file.write('Epoch: 0' + str(epoch) + ' Loss: ' + str(loss) + '\n')

                else:
                    test_output_file.write('Epoch: ' + str(epoch) + ' Loss: ' + str(loss) + '\n')

                if i % 100 == 0:    #in this case, this will print 2 images each testing cycle
                    plt.subplot(1, 4, 1)
                    plt.axis('off')
                    image = images[0].detach().squeeze().cpu().numpy()
                    plt.imshow(image)

                    plt.subplot(1, 4, 2)
                    plt.axis('off')
                    output = outputs[0].detach().squeeze().cpu().numpy()
                    plt.imshow(output)

                    plt.subplot(1, 4, 3)
                    plt.axis('off')
                    mask = masks[0].detach().squeeze().cpu().numpy()
                    plt.imshow(mask)

                    plt.subplot(1, 4, 4)
                    plt.axis('off')
                    plt.imshow(np.subtract(output, mask))
                    
                    plt.show()

                    
                    

            torch.save(model, MODEL_WEIGHTS_SAVE_PATH + str(epoch + 1) + '.pt')
        
    train_output_file.close()
    test_output_file.close()
    print('done')
    

train()