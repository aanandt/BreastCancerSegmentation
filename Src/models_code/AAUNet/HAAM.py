import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import pdb

class conv_block(nn.Module):

    def __init__(self,block_type,in_channels,out_channels,kernel_size,strides,padding,dilation, batch_norm,activation):

        super(conv_block,self).__init__()

        self.block = block_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.batch_norm = batch_norm
        self.activation = activation
        self.dilation = dilation


        if(self.block == 'conv' ):

            self.conv_1 = (nn.Conv2d(self.in_channels,self.out_channels
                ,self.kernel_size,self.strides,self.padding,self.dilation))

        elif(self.block == 'conv_T'  ):
            self.conv_1 = nn.ConvTranspose2d(self.in_channels,self.out_channels,self.kernel_size
                ,self.strides,self.padding)

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

            x = self.activation_fun(self.bn_1(self.conv_1(x)))

        else:

            x = self.activation_fun(self.conv_1(x))

        return x

class ChannelAttention(nn.Module):

	def __init__(self, in_channels, out_channels):
		super().__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		
		
		self.conv_5x5_block = conv_block('conv', self.in_channels, self.out_channels, (5,5), 1, 2, 1, True, 'LeakyReLu' )
		self.conv_3x3_block = conv_block('conv', self.in_channels, self.out_channels, (3,3), 1, 3, 3, True, 'LeakyReLu' )
		self.conv_1x1_block = conv_block('conv', self.out_channels*2, self.out_channels, (1,1), 1, 0, 1, True, 'LeakyReLu' )
		self.GAP = nn.AdaptiveAvgPool2d(1)

		self.dense1 = nn.Linear(self.out_channels*2, self.out_channels)
		self.dense2 = nn.Linear(self.out_channels, self.out_channels)

		self.sig = nn.Sigmoid()
		
	def forward(self, x):

		
		x_5x5 = self.conv_5x5_block(x)
		x_3x3_D = self.conv_3x3_block(x)
		x_GAP = self.GAP(torch.cat((x_5x5, x_3x3_D), 1))
		x_GAP = x_GAP.reshape(x_GAP.shape[0], x_GAP.shape[1])
		fc1= self.dense1(x_GAP)
		alpha = self.sig(self.dense2(fc1))
		alpha1 = 1 - alpha
		
		alpha = alpha.reshape(alpha.shape[0], alpha.shape[1], 1, 1)
		alpha1 = alpha1.reshape(alpha1.shape[0], alpha1.shape[1], 1, 1)
		
		x_5x5 = torch.mul(x_5x5, alpha1)
		x_3x3_D = torch.mul(x_3x3_D, alpha)
		out = self.conv_1x1_block(torch.cat((x_5x5, x_3x3_D), 1))
		
		return out

class SpatialAttention(nn.Module):
	
	def __init__(self, in_channels, out_channels):
		
		super().__init__()
		
		self.in_channels = in_channels
		self.out_channels = out_channels

		self.conv_3x3_block = conv_block('conv', self.in_channels, self.out_channels, (3,3), 1, 1, 1, True, 'LeakyReLu' )
		self.conv1_1x1 = conv_block('conv', self.out_channels, self.out_channels, (1,1), 1, 0, 1, True, 'LeakyReLu' )
		self.conv2_1x1 = conv_block('conv', self.out_channels*2, 1, (1,1), 1, 0, 1, True, 'LeakyReLu' )
		self.conv3_1x1 = conv_block('conv', self.out_channels*2, self.out_channels, (1,1), 1, 0, 1, True, 'LeakyReLu' )
		self.relu = nn.ReLU(inplace = True)
		self.sig = nn.Sigmoid()
		
	
	def forward(self, x, ch_attn_out):

		x_3x3 = self.conv1_1x1(self.conv_3x3_block(x))
		concat = self.relu(torch.cat((ch_attn_out, x_3x3), dim=1))
		beta = self.sig(self.conv2_1x1(concat))
		beta1 = 1 - beta
		beta = beta.repeat(1, self.out_channels, 1, 1)
		beta1 = beta1.repeat(1, self.out_channels, 1, 1)
		
		x_5x5 = torch.mul(x_3x3, beta1)
		x_3x3_D = torch.mul(ch_attn_out, beta)
		out = self.conv3_1x1(torch.cat((x_5x5, x_3x3_D), dim=1))
		
		return out

class HAAM_block(nn.Module):

	def __init__(self, in_channels, out_channels):
		
		super().__init__()
		
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.channel_attention_block = ChannelAttention(self.in_channels, self.out_channels)
		self.spatial_attention_block = SpatialAttention(self.in_channels, self.out_channels)

	def forward(self, x):
		ch_attn_out = self.channel_attention_block(x)
		out = self.spatial_attention_block(x, ch_attn_out)
		
		return out

