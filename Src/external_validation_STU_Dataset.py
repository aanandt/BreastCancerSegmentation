import pandas as pd
import pdb
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision 
import torchvision.datasets as datasets
from torch.utils.data import DataLoader,Dataset,random_split
import torchvision.transforms as transforms
import sys
import torchvision.utils as vutils
import torch.autograd as autograd
import time
from tqdm import tqdm
from torch.autograd import Variable
import matplotlib
import torchmetrics

from configuration import * 
from models_code.UNet.UNet import *
from models_code.UNet_plus.UNet3Plus import *
from models_code.UNet_plus.UNet2Plus import *
from models_code.Attention_UNet.Attention_UNet import *
from models_code.SegNet.SegNet_git import *
from models_code.AAUNet.AAUNet import *
from models_code.GCN_UNet.GCN_UNet import *

from data_loaders.data_loader_train_val_with_transforms import *
from utils import *
from models_code.UNet_plus.model_loss import *
from utils import *
transform = transforms.Compose(
	[
	# transforms.CenterCrop(600), 
	transforms.Resize((256,256),2),
	 transforms.ToTensor(),
	 transforms.Normalize((0.5,),(0.5,))])

transform_seg = transforms.Compose(
	[
	# transforms.CenterCrop(600),
	transforms.Resize((256,256),2),
	transforms.Grayscale(num_output_channels=1),
	transforms.ToTensor(),
	])

def evaluate_model(model, validation_dataloader, save_path, fold_id):

	MIOU = torchmetrics.JaccardIndex(task = 'multiclass',num_classes = 2,average = 'none').to(device)
	save_map_path_pred = os.path.join(save_path, str(fold_id), 'Pred')
	save_map_path_gt = os.path.join(save_path, str(fold_id), 'GT')
	save_map_path_org = os.path.join(save_path, str(fold_id), 'Image')
	os.makedirs(save_path,exist_ok=True)
	os.makedirs(save_map_path_pred,exist_ok=True)
	os.makedirs(save_map_path_gt,exist_ok=True)
	os.makedirs(save_map_path_org,exist_ok=True)
	
	f_1 = open(os.path.join(save_path, 'values.txt'),'a')

	f_1.write('\n This is the segmentation evaluation scores\n')
	f_1.flush()

	with torch.no_grad():
		val_loss = 0
		model.eval()

		dice_score_list = []
		val_loss_list =[]
		pred_list = []
		image_list = []
		gt_list = []
		acc = []
		mean_iou = []
		iou_c = []
		dice_score = []
		count_check= 0
		

		for data in tqdm((val_dataloader )):#, desc=f'Epoch {epoch + 1}/ {num_epochs} - Validation'):

			# image, mask, img_name, mask_name = data
			_, image, mask, img_name, mask_name = data

			image = image.to(device)
			mask = mask.to(device)
			
			out = model(image)
			out = (out>0.5).float()
			mask = (mask>0.5).float()
			loss = criterion(out, mask)
			val_loss += loss.item() * image.size(0)
			val_loss_list.append(val_loss)

			for i in range(len(out)):
				save_file (os.path.join(save_map_path_pred, mask_name[i].split('/')[-1]), out[i,0,:,:])
				save_file (os.path.join(save_map_path_gt, mask_name[i].split('/')[-1]), mask[i,0,:,:])
				save_file (os.path.join(save_map_path_org, mask_name[i].split('/')[-1]), image[i,0,:,:])

			#pdb.set_trace()
			if(count_check == 0):
				count_check += 1
				target = mask
				pred = out

			else:
				target = torch.cat((target,mask),dim = 0)
				pred = torch.cat((pred,out),dim = 0)


			acc_val  = multi_acc(out,mask)
			acc.append(acc_val)
			
		pred_list.append(out)
		gt_list.append(mask)
		image_list.append(image )
	
	avg_val_loss = sum(val_loss_list) / len(val_loss_list)
	preds = torch.cat((pred_list))
	masks = torch.cat((gt_list),)
	imgs = torch.cat((image_list))
	
	iou = MIOU(pred,target)
	avg = torch.mean(iou)
	iou_c_avg = iou[1]
	avg_dice = (2*iou_c_avg /(iou_c_avg+1))
	avg_p = sum(acc) / len(acc)

		
	acc, sen, spec, prec, iou, dice = get_score(pred, target)
	#pdb.set_trace()
	print('MIOU--->',avg,'--IOU-->',iou,'--DICE-->',dice,'--Acc-->',acc)
	print('--Sen-->',sen, '--Prec-->', prec, '--Spec-->',spec)
	f_1.write('--->MIOU--->'+str(avg)+'--IOU-->'+str(iou_c_avg)+'--DICE-->'+str(avg_dice)+'--Acc-->'+str(avg_p)+'--Sen-->'+str(sen)+'--Prec-->'+str(prec)+'--Spec-->'+str(spec)+'\n')
	f_1.flush()

def choose_model(model_name, num_gpus):
	if (model_name == 'AAUNet'):
		model = AAUNet()
		
	elif (model_name == 'Attention_UNet'):
		model = AttU_Net()

	elif (model_name == 'UNet'):
		model = UNet()

	elif (model_name == 'GCN_UNet'):
		model = GCN_UNet()

	elif (model_name == 'SegNet'):
		model = SegNet()

	elif (model_name == 'UNet2Plus'):
		model = UNet2Plus()

	elif (model_name == 'UNet3Plus'):
		model =  UNet3Plus()

	model.apply(reset_weights)
	model = model.to(device)
	if num_gpus == 2:
		model = nn.DataParallel(model,device_ids = [0,1])
		
	model.load_state_dict(torch.load(model_path)['net'])
	return model

config = configuration()
device = config.device 
model_name = config.model_name
num_gpus = config.num_gpus
k_fold = config.k_fold

map_save_path = 'Segmentation results folder' 
temp_model_path = 'Best model path saved for each fold_id'

train_score = pd.Series()
val_score = pd.Series()

lr_rate = 0.0001
for fold_id in range(k_fold):
	
	dataset = SegmentationDataset(data_dir='../Dataset/STU-Hospital-master/',transform_1 = transform, transform_2 = transform_seg)

	val_dataloader = torch.utils.data.DataLoader(dataset, batch_size=4,
									  shuffle=True, num_workers=4)


	model_file = [x for x in os.listdir(os.path.join(temp_model_path, str(fold_id))) if x.endswith('.pt')]
	model_path = os.path.join(temp_model_path, str(fold_id), str(model_file[0]))
	model = choose_model(model_name, num_gpus, model_path)
	

	model = model.to(device)
	 
	optimizer = optim.Adam(model.parameters(), lr = lr_rate)
	criterion = nn.BCELoss().to(device)

	evaluate_model(model, val_dataloader, map_save_path, fold_id)
pdb.set_trace()




