import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import pdb
from .HAAM import *

class AAUNet_block_down(nn.Module):
	def __init__(self,in_channels,out_channels):

		super().__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.l1 = HAAM_block(self.in_channels, self.out_channels)
		self.l2 = HAAM_block(self.out_channels, self.out_channels)

	def forward(self,x):
	   
		x = self.l2(self.l1(x))
		return x

class AAUnet_block_up(nn.Module):

	def __init__(self,in_channels,out_channels):

		super().__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels

		self.l1 = HAAM_block(self.in_channels, self.out_channels)
		self.l2 = HAAM_block(self.out_channels, self.out_channels)

	def forward(self,x):
	   
		x = self.l2(self.l1(x))
		return x


class AAUNet_encoder(nn.Module):

	def __init__(self):
		super().__init__()

		self.pool = nn.MaxPool2d((2,2),2)
		self.block_1 = AAUNet_block_down(3, 32)

		self.block_2 = AAUNet_block_down(32, 64)
		self.block_3 = AAUNet_block_down(64, 128)
		self.block_4 = AAUNet_block_down(128, 256)
		

	def forward(self, x):
		
		x1 = self.block_1(x)
		x2 = self.block_2(self.pool(x1))
		x3 = self.block_3(self.pool(x2))
		x4 = self.block_4(self.pool(x3))
		

		return x1, x2, x3, x4

class AAUNet(nn.Module):

	def __init__(self):
		super().__init__()

		self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
		self.pool = nn.MaxPool2d((2,2),2)
		self.encoder =  AAUNet_encoder()

		# This is the bottle_neck
		self.block_6 = AAUNet_block_down(256,512)

		self.up_1 = nn.ConvTranspose2d(512,256,(4,4),2,1)
		self.up_block_1 = AAUnet_block_up(256*2,256)

		self.up_2 = nn.ConvTranspose2d(256,128,(4,4),2,1)
		self.up_block_2 = AAUnet_block_up(128*2,128)

		self.up_3 = nn.ConvTranspose2d(128,64,(4,4),2,1)
		self.up_block_3 = AAUnet_block_up(64*2,64)

		self.up_4 = nn.ConvTranspose2d(64,32,(4,4),2,1)
		self.up_block_4 = AAUnet_block_up(32*2,32)

		self.pred = nn.Conv2d(32,1,(3,3),1,1)

	def forward(self,x):

		x1, x2, x3, x4 = self.encoder(x)
		

		x = self.pool(self.block_6(x4))


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


