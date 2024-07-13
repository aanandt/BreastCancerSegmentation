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
from models_code.AAUNet.AAUNet import *
from models_code.Attention_UNet.Attention_UNet import *
from models_code.GCN_UNet.GCN_UNet import *
from models_code.segnet.SegNet import *
from models_code.UNet_plus.UNet2Plus import *
from models_code.UNet_plus.UNet3Plus import *
from models_code.UNet.UNet import *


from data_loaders.data_loader_train_val_with_transforms import *
from utils import *
from models_code.AAUNet.model_loss import *

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

def get_train_val_dataset_fold(dataset, fold_id):

	#Divide the dataset into train and validation with fold_id

	total_size = len(dataset)
	fraction = 1/k_fold
	seg = int(total_size * fraction)

	trll = 0
	trlr = fold_id * seg
	vall = trlr
	valr = fold_id * seg + seg
	trrl = valr
	trrr = total_size
	
	train_left_indices = list(range(trll,trlr))
	train_right_indices = list(range(trrl,trrr))
	
	train_indices = train_left_indices + train_right_indices
	val_indices = list(range(vall,valr))
	
	
	train_set = torch.utils.data.dataset.Subset(dataset,train_indices)
	val_set = torch.utils.data.dataset.Subset(dataset,val_indices)

	return train_set, val_set

def train_model(train_dataloader_benign, train_dataloader_malignant, 
	train_dataloader_normal, validation_dataloader, model, criterion, 
	optimizer, save_path_model, save_path_map, fold_id, model_name, num_epochs):

	
	dice_min = 0
	save_path_model = os.path.join(save_path_model, str(fold_id))
	os.makedirs(save_path_model,exist_ok=True)

	f_1 = open(os.path.join(save_path_model, 'values.txt'),'a')

	f_1.write('\n This is the segmentation with ' + str(model_name) +'\n')
	f_1.flush()
	MIOU = torchmetrics.JaccardIndex(task = 'multiclass',num_classes = 2,average = 'none').to(device)
	save_map_path_pred = os.path.join(save_path_map, str(fold_id), 'Pred')
	save_map_path_gt = os.path.join(save_path_map, str(fold_id), 'GT')
	save_map_path_org = os.path.join(save_path_map, str(fold_id), 'Image')
	os.makedirs(save_path_map,exist_ok=True)
	
	os.makedirs(save_map_path_pred,exist_ok=True)
	os.makedirs(save_map_path_gt,exist_ok=True)
	os.makedirs(save_map_path_org,exist_ok=True)

	dataloader_malign_iterator = iter(train_dataloader_malignant)
	dataloader_normal_iterator = iter(train_dataloader_normal)
	for epoch in range(num_epochs):
		model.train()
		with torch.no_grad():
			bce_loss = 0
			train_loss = 0
			val_loss = 0
		for data in (tqdm((train_dataloader_benign), desc=f'Epoch {epoch + 1}/ {num_epochs} - Training')):

			_,image_b, mask_b, img_name_b, mask_name_b = data
			# stratified sampling
			
			try:
				_,image_m, mask_m, img_name_m, mask_name_m = next(dataloader_malign_iterator)
			except StopIteration:
				dataloader_malign_iterator = iter(train_dataloader_malignant)
				_,image_m, mask_m, img_name_m, mask_name_m = next(dataloader_malign_iterator)

			try:
				_,image_n, mask_n, img_name_n, mask_name_n = next(dataloader_normal_iterator)
			except StopIteration:
				dataloader_normal_iterator = iter(train_dataloader_normal)
				_,image_n, mask_n, img_name_n, mask_name_n = next(dataloader_normal_iterator)

			idx = torch.randperm(image_m.shape[0] + image_b.shape[0] + image_n.shape[0])
			images = combine_randperm(image_b,image_m,image_n,idx)
			masks = combine_randperm(mask_b,mask_m,mask_n,idx)

			images = images.to(device)
			masks = masks.to(device)

			# Forward pass
			optimizer.zero_grad()

			output = model(images)
			#pdb.set_trace()
			# Compute loss
			loss = criterion(output, masks)

			train_loss += loss.item() * images.size(0)

			loss.backward()
			optimizer.step()
			#break
		# Compute average training loss for the epoch
		avg_train_loss = train_loss / len(train_dataloader_benign.dataset)
		with torch.no_grad():
		
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
			count_check = 0
			for index in range(len(validation_dataloader)):

				val_dataloader = validation_dataloader[index]

				for data in tqdm((val_dataloader ), desc=f'Epoch {epoch + 1}/ {num_epochs} - Validation'):

					_,image, mask, img_name, mask_name = data

					image = image.to(device)
					mask = mask.to(device)
					# pdb.set_trace()
					out = model(image)
					out = (out>0.5).float()
					mask = (mask>0.5).float()
					loss = criterion(out, mask)
					val_loss += loss.item() * image.size(0)
					val_loss_list.append(val_loss)

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
		preds = torch.cat((pred_list[0], pred_list[1], pred_list[2]),dim=0)
		masks = torch.cat((gt_list[0], gt_list[1], gt_list[2]),0)
		imgs = torch.cat((image_list[0], image_list[1], image_list[2]),dim=0)
		
		iou = MIOU(pred,target)
		avg = torch.mean(iou)
		iou_c_avg = iou[1]
		avg_dice = (2*iou_c_avg /(iou_c_avg+1))
		avg_p = sum(acc) / len(acc)

		if(dice_min < avg_dice):
			dice_min = avg_dice
			state = {
					'net' : model.state_dict(),
					'net_opt' : optimizer.state_dict(),
					
					}
			torch.save(state,os.path.join(save_path_model, str(epoch)+'_state_UNet.pt'))

		
		save_map(masks.float(),os.path.join(save_map_path_gt, str(epoch)))
		save_map(preds.float(),os.path.join(save_map_path_pred, str(epoch)))
		save_map(imgs.float(),os.path.join(save_map_path_org, str(epoch)))
		
		acc, sen, spec, prec, iou, dice = get_score(pred, target)
		
		print('MIOU--->',avg,'--IOU-->',iou_c_avg,'--DICE-->',avg_dice,'--Acc-->',avg_p)
		print('--Sen-->',sen, '--Prec-->', prec, '--Spec-->',spec)
		f_1.write(str(epoch)+'--->MIOU--->'+str(avg)+'--IOU-->'+str(iou_c_avg)+'--DICE-->'+str(avg_dice)+'--Acc-->'+str(avg_p)+'--Sen-->'+str(sen)+'--Prec-->'+str(prec)+'--Spec-->'+str(spec)+'\n')
		f_1.flush()
	return avg_train_loss

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
	
	return model



config = configuration()
device = config.device 
model_name = config.model_name
num_gpus = config.num_gpus
temp_models_save_path = config.model_path
temp_map_save_path = config.map_path
num_epochs = config.num_epochs
k_fold = config.k_fold

models_save_path = os.path.join(temp_models_save_path, model_name)
map_save_path = os.path.join(temp_map_save_path, model_name)


train_score = pd.Series()
val_score = pd.Series()

lr_rate = 0.0001

for fold_id in range(k_fold):
	
	dataset_benign = SegmentationDataset(data_dir='../Dataset/BUSI/benign',transform_1 = transform, transform_2 = transform_seg)
	dataset_malignant = SegmentationDataset(data_dir='../Dataset/BUSI/malignant',transform_1 = transform,transform_2 = transform_seg)
	dataset_normal = SegmentationDataset(data_dir='../Dataset/BUSI/normal',transform_1 = transform,transform_2 = transform_seg)
	
	train_benign, val_benign = get_train_val_dataset_fold(dataset_benign, fold_id)
	train_malignant, val_malignant = get_train_val_dataset_fold(dataset_malignant, fold_id)
	train_normal, val_normal = get_train_val_dataset_fold(dataset_normal, fold_id)

	train_dataloader_benign = torch.utils.data.DataLoader(train_benign, batch_size=2,
										  shuffle=True, num_workers=4)
	val_dataloader_benign = torch.utils.data.DataLoader(val_benign, batch_size=4,
									  shuffle=True, num_workers=4)

	train_dataloader_malignant = torch.utils.data.DataLoader(train_malignant, batch_size=2,
										  shuffle=True, num_workers=4)
	val_dataloader_malignant = torch.utils.data.DataLoader(val_malignant, batch_size=4,
									  shuffle=True, num_workers=4)
	
	train_dataloader_normal = torch.utils.data.DataLoader(train_normal, batch_size=4,
										  shuffle=True, num_workers=4)
	val_dataloader_normal = torch.utils.data.DataLoader(val_normal, batch_size=4,
									  shuffle=True, num_workers=4)
	validation_dataloader = []
	validation_dataloader.append(val_dataloader_benign)
	validation_dataloader.append(val_dataloader_malignant)
	validation_dataloader.append(val_dataloader_normal)

	
	
	model = choose_model(model_name, num_gpus)

	optimizer = optim.Adam(model.parameters(), lr = lr_rate)
	criterion = nn.BCELoss().to(device)

	train_acc = train_model(train_dataloader_benign, train_dataloader_malignant, 
				train_dataloader_normal, validation_dataloader, model, criterion, 
				optimizer, models_save_path, map_save_path, fold_id, model_name, num_epochs)
	
	train_score.at[fold_id] = train_acc
	




