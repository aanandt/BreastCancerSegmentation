import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import sys
import pdb
import torchvision.models as models\

def initialize_weights(*models):
	for model in models:
		for m in model.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1.)
				m.bias.data.fill_(1e-4)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0.0, 0.0001)
				m.bias.data.zero_()

class GCN_Block(nn.Module):

	def __init__(self,in_channels,out_channels,k=7): #out_Channel=21 in paper

		super(GCN_Block, self).__init__()

		self.conv_l1 = nn.Conv2d(in_channels, out_channels, (k,1), 1,  (int((k-1)/2),0))
		self.conv_l2 = nn.Conv2d(out_channels, out_channels, (1,k), 1, (0,int((k-1)/2)))
		self.conv_r1 = nn.Conv2d(in_channels, out_channels, (1,k), 1, (int((k-1)/2),0))
		self.conv_r2 = nn.Conv2d(out_channels, out_channels, (k,1), 1, (0,int((k-1)/2)))
		self.bn = nn.InstanceNorm2d(out_channels)
		self.activation_fun = nn.LeakyReLU(0.2,inplace = True)
	def forward(self, x):
		x_l = self.conv_l1(x)
		x_l = self.conv_l2(x_l)
		
		x_r = self.conv_r1(x)
		x_r = self.conv_r2(x_r)
		
		x = x_l + x_r
		x = self.activation_fun(self.bn(x))
		return x

# class BR_Block(nn.Module):
# 	def __init__(self, out_channels):
# 		super(BR_Block, self).__init__()
# 		self.bn = nn.BatchNorm2d(out_channels)
# 		self.relu = nn.ReLU(inplace=True)
# 		self.conv1 = nn.Conv2d(out_channels,out_channels, (3,3), 1, 1)
# 		self.conv2 = nn.Conv2d(out_channels,out_channels, (3,3), 1, 1)
	
# 	def forward(self,x):
# 		x_res = x
# 		#x_res = self.bn(x)
# 		#x_res = self.relu(x_res)
# 		x_res = self.conv1(x_res)
# 		#x_res = self.bn(x_res)
# 		x_res = self.relu(x_res)
# 		x_res = self.conv2(x_res)
		
# 		x = x + x_res
		
# 		return x



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
		
			self.bn_1 = nn.InstanceNorm2d(self.out_channels)
			

		if(activation == 'ReLu'):

			self.activation_fun = nn.ReLU(inplace = True)

		elif(activation == 'LeakyReLu'):

			self.activation_fun = nn.LeakyReLU(0.2,inplace = True)

		elif(activation == 'Swash'):

			self.activation_fun = swash()

		else:

			self.activation_fun = nn.LeakyReLU(negative_slope = 0.2,inplace = True)


	def forward(self,x):

		if(self.batch_norm == True):

			x = self.activation_fun(self.bn_1(self.conv_1(x)))

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

		self.l1 = conv_block('conv',in_channels,out_channels,(3,3),1,1,True,'LeakyReLu')
		self.l2 = conv_block('conv',out_channels,out_channels,(3,3),1,1,True,'LeakyReLu')

	def forward(self,x):

		x = self.l2(self.l1(x))

		return x

class GCN_UNet_block_down(nn.Module):
	def __init__(self,in_channels,out_channels):

		super().__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.l1 = GCN_Block(in_channels,out_channels)
		self.l2 = GCN_Block(out_channels,out_channels)
		self.l3 = nn.Conv2d(in_channels,out_channels,(1,1),1,0)
		self.dropout = nn.Dropout(0.2) 
	def forward(self,x):
		#x1 = self.l3(x)

		x = self.l2(self.l1(x))
		#x = x + x1
		#x = self.dropout(x + x1)
		return x

class GCN_Unet_block_up(nn.Module):

	def __init__(self,in_channels,out_channels):

		super().__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.l1 = GCN_Block(in_channels,out_channels)
		self.l2 = GCN_Block(out_channels,out_channels)

		self.l3 = nn.Conv2d(in_channels,out_channels,(1,1),1,0)
		self.dropout = nn.Dropout(0.2)
	def forward(self,x):
				
		#x1 = self.l3(x)

		x = self.l2(self.l1(x))
		#x = x + x1
		# x = self.dropout(x + x1)
		
		return x


class GCN_UNet_encoder(nn.Module):

	def __init__(self):
		super().__init__()

		self.pool = nn.MaxPool2d((2,2),2)
		self.block_1 = GCN_UNet_block_down(3, 32)

		self.block_2 = GCN_UNet_block_down(32, 64)
		self.block_3 = GCN_UNet_block_down(64, 128)
		self.block_4 = GCN_UNet_block_down(128, 256)
		self.block_5 = GCN_UNet_block_down(256, 512)

	def forward(self, x):
		#pdb.set_trace()
		x1 = self.block_1(x)
		x2 = self.block_2(self.pool(x1))
		x3 = self.block_3(self.pool(x2))
		x4 = self.block_4(self.pool(x3))
		x5 = self.block_5(self.pool(x4))

		return x1, x2, x3, x4, x5

class GCN_UNet(nn.Module):

	def __init__(self):
		super().__init__()

		self.pool = nn.MaxPool2d((2,2),2)
		self.encoder_UNet =  GCN_UNet_encoder()

		# This is the bottle_neck
		self.block_6 = GCN_UNet_block_down(512,1024)
		
		self.up_6 = nn.ConvTranspose2d(1024,512,(4,4),2,1)
		self.up_block_6 = GCN_Unet_block_up(512*2,512)		

		self.up_5 = nn.ConvTranspose2d(512,256,(4,4),2,1)
		self.up_block_5 = GCN_Unet_block_up(256*2,256)

		self.up_4 = nn.ConvTranspose2d(256,128,(4,4),2,1)
		self.up_block_4 = GCN_Unet_block_up(128*2,128)	

		self.up_3 = nn.ConvTranspose2d(128,64,(4,4),2,1)
		self.up_block_3 = GCN_Unet_block_up(64*2,64)		

		self.up_2 = nn.ConvTranspose2d(64,32,(4,4),2,1)
		self.up_block_2 = GCN_Unet_block_up(32*2,32)

		self.pred = nn.Conv2d(32,1,(3,3),1,1)

	def forward(self,x):

		x1, x2, x3, x4, x5 = self.encoder_UNet(x)
		x6 = self.pool(self.block_6(x5))
	
		x = self.up_6(x6)
		x = self.up_block_6(torch.cat((x,x5),dim = 1))

		x = self.up_5(x)
		x = self.up_block_5(torch.cat((x,x4),dim = 1))

		x = self.up_4(x)
		x = self.up_block_4(torch.cat((x,x3),dim = 1))

		x = self.up_3(x)
		x = self.up_block_3(torch.cat((x,x2),dim = 1))

		x = self.up_2(x)
		x = self.up_block_2(torch.cat((x,x1),dim = 1)) 
		
		x = self.pred(x).sigmoid()

		return x

