import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import sys
import pdb
import torchvision.models as models

class conv_block(nn.Module):

	def __init__(self,block_type,in_channels,out_channels,kernel_size,strides,padding,batch_norm,activation):

		super(conv_block,self).__init__()

		self.block = block_type
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.strides = strides
		self.padding = padding
		self.batch_norm = batch_norm
		self.activation = activation

		if(self.block == 'conv'):

			self.conv_1 = (nn.Conv2d(self.in_channels,self.out_channels
				,self.kernel_size,self.strides,self.padding))

		elif(self.block == 'conv_T'):

			self.conv_1 = (nn.ConvTranspose2d(self.in_channels,self.out_channels
				,self.kernel_size,self.strides,self.padding))    


		if(self.batch_norm == True):

			self.bn_1 = nn.BatchNorm2d(self.out_channels)
			

		if(activation == 'ReLu'):

			self.activation_fun = nn.ReLU(inplace = True)
		
		elif(activation == 'LeakyReLu'):

			
			self.activation_fun = nn.LeakyReLU(0.01,inplace = True)

		elif(activation == 'Swash'):

			self.activation_fun = swash()

		else:

			self.activation_fun = nn.LeakyReLU(negative_slope = 0.2,inplace = True)


	def forward(self,x):

		if(self.batch_norm == True):

			x = self.activation_fun((self.conv_1(x)))

		else:

			x = self.activation_fun(self.conv_1(x))

		return x

class UNet_block(nn.Module):

	def __init__(self,in_channels,out_channels):

		super().__init__()  

		self.in_channels = in_channels
		self.out_channels = out_channels

		# if(in_channels != out_channels):
		#     self.l3 = nn.Conv2d(in_channels,out_channels,(1,1),1,0)

		self.l1 = conv_block('conv',in_channels,out_channels,(3,3),1,1,True,'ReLu')
		self.l2 = conv_block('conv',out_channels,out_channels,(3,3),1,1,True,'ReLu')

	def forward(self,x):

		x = self.l2(self.l1(x))

		return x

class UNet_block_down(nn.Module):
	def __init__(self,in_channels,out_channels):

		super().__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.l1 = conv_block('conv',in_channels,out_channels,(3,3),1,1,True,'ReLu')
		self.l2 = conv_block('conv',out_channels,out_channels,(3,3),1,1,True,'ReLu')

	def forward(self,x):
	   
		x = self.l2(self.l1(x))
		return x

class Unet_block_up(nn.Module):

	def __init__(self,in_channels,out_channels):

		super().__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.l1 = conv_block('conv',in_channels,out_channels,(3,3),1,1,True,'ReLu')
		self.l2 = conv_block('conv',out_channels,out_channels,(3,3),1,1,True,'ReLu')

	def forward(self,x):
	   
		x = self.l2(self.l1(x))
		return x


class UNet_encoder(nn.Module):

	def __init__(self):
		super().__init__()

		self.pool = nn.MaxPool2d((2,2),2)
		self.block_1 = UNet_block_down(3, 32)

		self.block_2 = UNet_block_down(32, 64)
		self.block_3 = UNet_block_down(64, 128)
		self.block_4 = UNet_block_down(128, 256)
		self.block_5 = UNet_block_down(256, 512)

	def forward(self, x):
		#pdb.set_trace()
		x1 = self.block_1(x)
		x2 = self.block_2(self.pool(x1))
		x3 = self.block_3(self.pool(x2))
		x4 = self.block_4(self.pool(x3))
		x5 = self.block_5(self.pool(x4))

		return x1, x2, x3, x4,x5

class UNet(nn.Module):

	def __init__(self):
		super().__init__()

		self.pool = nn.MaxPool2d((2,2),2)
		self.encoder_UNet =  UNet_encoder()

		# This is the bottle_neck
		self.block_6 = UNet_block_down(512,1024)


		self.up_0 = nn.ConvTranspose2d(1024,512,(4,4),2,1)
		self.up_block_0 = Unet_block_up(512*2,512)

		self.up_1 = nn.ConvTranspose2d(512,256,(4,4),2,1)
		self.up_block_1 = Unet_block_up(256*2,256)

		self.up_2 = nn.ConvTranspose2d(256,128,(4,4),2,1)
		self.up_block_2 = Unet_block_up(128*2,128)

		self.up_3 = nn.ConvTranspose2d(128,64,(4,4),2,1)
		self.up_block_3 = Unet_block_up(64*2,64)

		self.up_4 = nn.ConvTranspose2d(64,32,(4,4),2,1)
		self.up_block_4 = Unet_block_up(32*2,32)

		self.pred = nn.Conv2d(32,1,(3,3),1,1)

	def forward(self,x):

		x1, x2, x3, x4, x5 = self.encoder_UNet(x)
		
		#pdb.set_trace()

		x = self.pool(self.block_6(x5))
		
		x = self.up_0(x)
		x = self.up_block_0(torch.cat((x,x5),dim = 1))

		x = self.up_1(x)
		x = self.up_block_1(torch.cat((x,x4),dim = 1))

		x = self.up_2(x)
		x = self.up_block_2(torch.cat((x,x3),dim = 1))

		x = self.up_3(x)
		x = self.up_block_3(torch.cat((x,x2),dim = 1))

		x = self.up_4(x)
		x = self.up_block_4(torch.cat((x,x1),dim = 1))

		x = self.pred(x).sigmoid()

		return x




