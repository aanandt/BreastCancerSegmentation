import numpy as np  
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import matplotlib
import torch
from sklearn.metrics import roc_curve, auc
import pdb  

def combine_randperm(images_1,images_2, images_3, idx):
	
	images = torch.cat((images_1, images_2, images_3),dim = 0)
	images = images[idx].view(images.size())

	return images
def save_img(images,path):

    device = 'cuda:0'
    z = np.transpose(vutils.make_grid(images.to(device),padding=2, normalize=True,nrow = 4).cpu(),(1,2,0))
    z = z.numpy()
    matplotlib.image.imsave(path+'.jpg', z)

    return 0

def save_map(images,path):

    device = 'cuda:0'
    # z = np.transpose(vutils.make_grid(fake_1_c.to(device)[:36],padding=2, normalize=True,nrow = 5).cpu(),(1,2,0))
    # z = z.numpy()
    # matplotlib.image.imsave(path_main_attn_64+'.jpg', z,cmap = 'jet'

    # print(images.shape)
    z = np.transpose(vutils.make_grid(images.to(device),padding=2, normalize=True,nrow = 4).cpu(),(1,2,0))
    #print(z.shape)
    
    z = z.numpy()
    z = z[:,:,0]
    matplotlib.image.imsave(path+'.jpg', z, cmap = 'gray')

    return 0

def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            #print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

def intersection_over_union(pred_mask, true_mask):
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    union = torch.logical_or(pred_mask, true_mask).sum()
    iou = intersection / union
    return iou

def dice_coefficient(pred_mask, true_mask):
    intersection = torch.logical_and(pred_mask, true_mask).sum()
    dice = (2. * intersection) / (pred_mask.sum() + true_mask.sum())
    return dice

def accuracy(pred_mask, true_mask):
    correct = (pred_mask == true_mask).sum()
    total = pred_mask.numel()
    acc = correct / total
    return acc

def precision(pred_mask, true_mask):
    true_positive = (pred_mask & true_mask).sum().float()
    false_positive = (pred_mask & ~true_mask).sum().float()
    precision = true_positive / (true_positive + false_positive + 1e-6)  # Adding epsilon to avoid division by zero
    return precision

def recall(pred_mask, true_mask):
    true_positive = (pred_mask & true_mask).sum().float()
    false_negative = (~pred_mask & true_mask).sum().float()
    recall = true_positive / (true_positive + false_negative + 1e-8)  # Adding epsilon to avoid division by zero
    return recall


def specificity(pred_mask, true_mask):
    true_negative = torch.logical_and(~pred_mask, ~true_mask).sum()
    false_positive = torch.logical_and(pred_mask, ~true_mask).sum()
    specificity = true_negative / (true_negative + false_positive)
    return specificity

def get_score(pred, target):
    
    pred_mask = pred.type(torch.bool)
    true_mask = target.type(torch.bool)

    acc = accuracy(pred_mask, true_mask)
    sen = recall(pred_mask, true_mask)
    spec = specificity(pred_mask, true_mask)
    prec = precision(pred_mask, true_mask)
    iou = intersection_over_union(pred_mask, true_mask)
    dice = dice_coefficient(pred_mask, true_mask)

    return (acc), (sen), (spec), (prec), (iou), (dice)


# def get_f1_score(pd, gt, threshold=0.5):
#     """
#     :param threshold:
#     :param pd: prediction
#     :param gt: ground truth
#     :return: dice coefficient or f1-score
#     """

#     pd = (pd > threshold).float()
#     intersection = torch.sum((pd + gt) == 2)

#     score = float(2 * intersection) / (float(torch.sum(pd) + torch.sum(gt)) + 1e-6)
#     return score


# def get_iou_score(pd, gt, threshold=0.5):
#     """
#     :param threshold:
#     :param pd: prediction
#     :param gt: ground truth
#     :return: iou score or jaccard similarity
#     """

#     pd = (pd > threshold).float()
#     intersection = torch.sum((pd + gt) == 2)
#     union = torch.sum((pd + gt) >= 1)

#     score = float(intersection) / (float(union) + 1e-6)
#     return score


# def get_accuracy(pd, gt, threshold=0.5):
#     """
#     formula = (tp + tn) / (tp + tn + fp + fn)
#     :param threshold:
#     :param pd: prediction
#     :param gt: ground truth
#     :return: accuracy score
#     """

#     pd = (pd > threshold).float()
#     corr = torch.sum(pd == gt)
#     tensor_size = pd.size(0) * pd.size(1) * pd.size(2) * pd.size(3)

#     score = float(corr) / float(tensor_size)
#     return score


# def get_sensitivity(pd, gt, threshold=0.5):
#     """
#     formula = tp / (tp + fn)
#     :param threshold:
#     :param pd: prediction
#     :param gt: ground truth
#     :return: sensitivity or recall rate
#     """

#     pd = (pd > threshold).float()
#     tp = (((pd == 1).float() + (gt == 1).float()) == 2).float()  # True Positive
#     fn = (((pd == 0).float() + (gt == 1).float()) == 2).float()  # False Negative

#     score = float(torch.sum(tp)) / (float(torch.sum(tp + fn)) + 1e-6)
#     return score


# def get_specificity(pd, gt, threshold=0.5):
#     """
#     formula = tn / (tn + fp)
#     :param threshold:
#     :param pd: prediction
#     :param gt: ground truth
#     :return: specificity score
#     """
#     pd = (pd > threshold).float()
#     tn = (((pd == 0).float() + (gt == 0).float()) == 2).float()  # True Negative
#     fp = (((pd == 1).float() + (gt == 0).float()) == 2).float()  # False Positive

#     score = float(torch.sum(tn)) / (float(torch.sum(tn + fp)) + 1e-6)
#     return score


# def get_precision(pd, gt, threshold=0.5):
#     """
#     formula = tp / (tp + fn)
#     :param threshold:
#     :param pd: prediction
#     :param gt: ground truth
#     :return: precision score
#     """
#     pd = (pd > threshold).float()
#     tp = (((pd == 1).float() + (gt == 1).float()) == 2).float()  # True Positive
#     fp = (((pd == 1).float() + (gt == 0).float()) == 2).float()  # False Positive

#     score = float(torch.sum(tp)) / (float(torch.sum(tp + fp)) + 1e-6)

#     return score


# def get_mae(pd, gt):
#     """
#     mean absolute error
#     :param pd: prediction
#     :param gt: ground truth
#     :return: mae score
#     """
#     pd = torch.flatten(pd)
#     gt = torch.flatten(gt)
#     score = torch.mean(torch.abs(pd - gt))

#     return score.item()


# def get_mse(pd, gt):
#     """
#     mean squared error
#     :param pd: prediction
#     :param gt: ground truth
#     :return: mse score
#     """
#     pd = torch.flatten(pd)
#     gt = torch.flatten(gt)
#     score = torch.mean((pd - gt) ** 2)

#     return score.item()


# def get_rmse(pd, gt):
#     """
#     root mean squared error
#     :param pd: prediction
#     :param gt: ground truth
#     :return: rmse score
#     """
#     pd = torch.flatten(pd)
#     gt = torch.flatten(gt)
#     score = torch.sqrt(torch.mean((pd - gt) ** 2))

#     return score.item()


# def get_auc(pd, gt):
#     fpr, tpr, _ = roc_curve(
#         gt.flatten().cpu().detach().numpy().astype(np.uint8),
#         pd.flatten().cpu().detach().numpy(),
#         pos_label=1,
#     )
#     score = auc(fpr, tpr)

#     return score


# def get_score(pd, gt):
 
#     acc = (get_accuracy(pred, mask))
#     sen = (get_sensitivity(pred, mask))
#     spec = (get_specificity(pred, mask))
#     prec = (get_precision(pred, mask))
#     iou = (get_iou_score(pred, mask))
#     dice = (get_f1_score(pred, mask))

#     return (acc), (sen), (spec), (prec), (iou), (dice)

# def get_score(pd, gt, mode='acc'):
#     if mode == 'acc':
#         return get_accuracy(pd, gt)
#     elif mode == 'se':
#         return get_sensitivity(pd, gt)
#     elif mode == 'sp':
#         return get_specificity(pd, gt)
#     elif mode == 'pr':
#         return get_precision(pd, gt)
#     elif mode == 'iou':
#         return get_iou_score(pd, gt)
#     elif mode == 'dc':
#         return get_f1_score(pd, gt)
#     # elif mode == 'mae':
#     #     return get_mae(pd, gt)
#     # elif mode == 'mse':
#     #     return get_mse(pd, gt)
#     # elif mode == 'rmse':
#     #     return get_rmse(pd, gt)
#     # elif mode == 'auc':
#     #     return get_auc(pd, gt)
#     else:
#         print('Please check the mode is available.')
#         exit(0)            