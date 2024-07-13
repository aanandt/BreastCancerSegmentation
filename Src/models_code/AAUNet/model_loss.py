import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import sys
import torch.nn.utils.spectral_norm  as spectral_norm
import torchvision.models as models
import torchvision.utils as vutils
import matplotlib

device = 'cuda:0'

# criterion_log_softmax = nn.LogSoftmax(dim=-1).to(device)
# criterion_nll = nn.NLLLoss(reduction = 'none').to(device)
# criterion_hinge_in = torch.nn.HingeEmbeddingLoss().to(device)
criterion = nn.BCELoss(reduction = 'none').to(device)
criterion_l2 = nn.MSELoss().to(device)
criterion_l1 = nn.L1Loss().to(device)

ALPHA = 0.25
BETA = 0.75

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
        
        return 1 - IoU


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

        self.iou = IoULoss()
        self.focal = FocalLoss()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss #+ self.iou(inputs,targets) #+ self.focal(inputs,targets)
        
        return Dice_BCE

def TNSCUI_preprocess(image,seg,gt):
    # image_path = r'/media/root/老王3号/challenge/tnscui2020_train/image/1273.PNG'
    img = Image.open(image_path)
    Transform = T.Compose([T.ToTensor()])
    img_tensor = Transform(img)
    img_dtype = img_tensor.dtype
    img_array_fromtensor = (torch.squeeze(img_tensor)).data.cpu().numpy()

    img_array = np.array(img, dtype=np.float32)



    or_shape = img_array.shape  #原始图片的尺寸


    value_x = np.mean(img, 1) #% 为了去除多余行，即每一列平均
    value_y = np.mean(img, 0) #% 为了去除多余列，即每一行平均

    x_hold_range = list((len(value_x) * np.array([0.24 / 3, 2.2 / 3])).astype(np.int))
    y_hold_range = list((len(value_y) * np.array([0.8 / 3, 1.8 / 3])).astype(np.int))
    # x_hold_range = list((len(value_x) * np.array([0.8 / 3, 2.2 / 3])).astype(np.int))
    # y_hold_range = list((len(value_y) * np.array([0.8 / 3, 2.2 / 3])).astype(np.int))

    # value_thresold = 0
    value_thresold = 5


    x_cut = np.argwhere((value_x<=value_thresold)==True)
    x_cut_min = list(x_cut[x_cut<=x_hold_range[0]])
    if x_cut_min:
        x_cut_min = max(x_cut_min)
    else:
        x_cut_min = 0

    x_cut_max = list(x_cut[x_cut>=x_hold_range[1]])
    if x_cut_max:
        # print('q')
        x_cut_max = min(x_cut_max)
    else:
        x_cut_max = or_shape[0]


    y_cut = np.argwhere((value_y<=value_thresold)==True)
    y_cut_min = list(y_cut[y_cut<=y_hold_range[0]])
    if y_cut_min:
        y_cut_min = max(y_cut_min)
    else:
        y_cut_min = 0

    y_cut_max = list(y_cut[y_cut>=y_hold_range[1]])
    if y_cut_max:
        # print('q')
        y_cut_max = min(y_cut_max)
    else:
        y_cut_max = or_shape[1]


    if orimg:
        x_cut_max = or_shape[0]
        x_cut_min = 0
        y_cut_max = or_shape[1]
        y_cut_min = 0
    # 截取图像
    cut_image = img_array_fromtensor[x_cut_min:x_cut_max,y_cut_min:y_cut_max]
    cut_image_orshape = cut_image.shape

    cut_image = resize(cut_image, (outputsize, outputsize), order=3)

    cut_image_tensor = torch.tensor(data = cut_image,dtype=img_dtype)

    return [cut_image_tensor, cut_image_orshape,or_shape,[x_cut_min,x_cut_max,y_cut_min,y_cut_max]]

def find_location(seg_map):

    curr_x_val = 0
    curr_y_val = 0
    mask_present = 0

    x_val = seg_map.mean(dim = 2)
    y_val = seg_map.mean(dim = 1)

    x_1s = x_val[0].nonzero(as_tuple=False)
    y_1s = y_val[0].nonzero(as_tuple=False)

    if(x_1s.nelement() == 0):
        
        return 0,0,0,0,0

    else:

        x_min = x_1s.min() 
        x_max = x_1s.max()

        y_min = y_1s.min() 
        y_max = y_1s.max()
        mask_present = 1

        return x_min,x_max,y_min,y_max,mask_present



def crop_images(image,seg,gt):

    if(seg.shape[0] != 1):
        seg = seg.unsqueeze(dim = 0)

    gt = gt.unsqueeze(dim = 0)

    x_min,x_max,y_min,y_max,mask_present = find_location(seg)

    if(mask_present == 0 or (x_min-x_max) == 0 or (y_min-y_max)== 0):

        return 0,0,0

    image_cut = image[:,x_min:x_max,y_min:y_max]
    seg_cut = gt[:,x_min:x_max,y_min:y_max]

    image_cut = image_cut.unsqueeze(dim = 0)
    seg_cut = seg_cut.unsqueeze(dim = 0)

    image_cut = F.interpolate(image_cut,[128,128],mode = 'bilinear')
    seg_cut = F.interpolate(seg_cut,[128,128],mode = 'nearest')

    return image_cut.detach(),seg_cut.detach(),1

# class DiceBCELoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceBCELoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         # inputs = F.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
#         BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#         Dice_BCE = BCE + dice_loss
        
#         return Dice_BCE

# def orthogonal_regularisation(net,reg = 1e-4):

#     # reg = 1e-4
#     orth_loss = torch.zeros(1,device = device)
#     for name, param in net.named_parameters():
#         if 'bias' not in name:
#             param_flat = param.view(param.shape[0], -1)
#             sym = torch.mm(param_flat, torch.t(param_flat))
#             eye_val = torch.ones(param_flat.shape[0],device = device) - torch.eye(param_flat.shape[0],device =device)
#             sym = sym * eye_val
#             sym = torch.norm(sym)
#             orth_loss = orth_loss + (reg * sym)

#     return orth_loss

# def criterion_hinge(value,train_type):

#     if(train_type == 'd_true'):

#         return (torch.nn.ReLU()(1.0 - value)).mean()

#     if(train_type == 'd_false'):

#         return (torch.nn.ReLU()(1.0 + value)).mean()

#     if(train_type == 'd_diff'):

#         return (torch.nn.ReLU()(1.0 + value)).mean()

#     if(train_type == 'g_train'):

#         return (-value.mean())

# criterion_cross = nn.CrossEntropyLoss()

criterion_cross = nn.CrossEntropyLoss(reduction = 'none').to(device)

def contrastive_loss_text(C_text,img_features,sent,word):

    scores_sent,scores_word = C_sent(img_features,sent,word)

    class_numbers = torch.arange(0,img_features.shape[0])
    labels = class_numbers.to(device)

    value_1 = criterion_cross(scores_sent,labels)
    value_2 = criterion_cross(scores_word,labels)

    value = value_1 + value_2

    return value

def contrastive_loss_img_main(real,fake):

    fake = fake.view([-1,512])
    real = real.view([-1,512])

    real_norm = torch.norm(real,2,dim = 1,keepdim = True)
    fake_norm = torch.norm(fake,2,dim = 1,keepdim = True)

    scores = torch.mm(real,fake.transpose(0,1))
    norm = torch.mm(real_norm,fake_norm.transpose(0,1))

    scores = scores/norm.clamp(min=1e-8)

    class_numbers = np.arange(0,real.shape[0])
    labels = torch.Tensor(class_numbers)
    labels = labels.type(torch.LongTensor)
    labels = labels.to(device)
    
    value = torch.mean(criterion_cross(scores,labels) + criterion_cross(scores.transpose(0,1),labels) )

    return value

class generate_img(nn.Module):

    def __init__(self,channels):

        super().__init__()

        layers = []

        start = 512
        for i in channels:
            layers.append(conv_block('conv_T',start,i,(4,4),2,1,False,True,'ReLu'))
            start = i

        self.conv_layer = nn.Sequential(*layers)
        self.final = nn.Conv2d(64,3,(1,1),1,0)

    def forward(self,x):

        x = torch.tanh(self.final(self.conv_layer(x)))

        return x

class gen_image_both(nn.Module):

    def __init__(self):

        super().__init__()

        self.l1 = generate_img([256,128,64])
        self.l2 = generate_img([256,128,64])

    def forward(self,x1,x2,r1,r2):

        x1 = self.l1(x1)
        x2 = self.l2(x2)

        v1 = criterion_l1(x1,r1)
        v2 = criterion_l2(x2,r2)

        return x1,x2,(v1+v2)

def abp_loss(fake,real):

    val = fake.shape[2]
    fake = fake.view([-1,fake.shape[1],fake.shape[2]**2])
    real = real.view([-1,real.shape[1],real.shape[2]**2])

    fake = fake.permute(0,2,1)

    m = torch.bmm(fake,real)
    m = F.softmax(m, dim=1)

    fake = fake.permute(0,2,1)

    fake = torch.bmm(fake,m)

    fake = fake.view([-1,512,8,8])

    # value = criterion_l2(fake,real)

    return fake

def contrastive_loss_img(C_img,r1,r2,f1,f2):

    scores_1,scores_2 = C_img(r1,r2,f1,f2)

    # class_numbers = torch.arange(0,r1.shape[0])
    # labels = class_numbers.to(device)

    class_numbers = np.arange(0,r1.shape[0])
    labels = torch.Tensor(class_numbers)
    labels = labels.type(torch.LongTensor)
    labels = labels.to(device)
    
    value_1 = (criterion_cross(scores_1,labels))
    value_2 = (criterion_cross(scores_2,labels))

    value_1 = torch.mean(value_1)
    value_2 = torch.mean(value_2)

    value = value_1 + value_2

    return value


def model_loss_contrastive_dist(x,y):

    sim = []

    for i in y:
        i = i.unsqueeze(0)
        i = i.repeat(x.shape[0],1)
        c = torch.abs(i-x)
        c = c.mean(-1,keepdim = True)
        sim.append(c)

    sim = torch.cat(sim,1)

    class_numbers = np.arange(0,x.shape[0])
    labels = torch.Tensor(class_numbers)
    labels = labels.type(torch.LongTensor)
    labels = labels.to(device)

    value = torch.mean(criterion_cross(sim,labels)) + torch.mean(criterion_cross(sim.transpose(0, 1),labels))

    return value

def model_loss_disc(netD,images,fake_1,train_type):

    
    a1,f = netD(images)
    a3,_ = netD(fake_1)
    # a4,_ = netD(fake_2)

    value_1 = (criterion_hinge(a1,'d_true'))
    value_2 = (criterion_hinge(a3,'d_false'))
    
    value = (value_1 + (value_2 ))

    return value,a1

def model_loss(netD,netD_i,fake_1,fake_m,train_type,labels):

    
    a1,f1 = netD(fake_1)
    f2 = netD_i(fake_m)

    value_1 = (criterion_hinge(a1,train_type))
    value_2 = torch.mean(criterion_cross(f2,labels))
    
    value = (value_1 +(value_2))

    return value


def contrastive_loss_inception(net,real,fake_1,fake_2):

    fake_1 = F.interpolate(fake_1,[299,299])
    fake_2 = F.interpolate(fake_2,[299,299])
    real = F.interpolate(real,[299,299])

    real,middle_features = net(real)
    fake_1,middle_features = net(fake_1)
    fake_2,middle_features = net(fake_2)

    class_numbers = np.arange(0,real.shape[0])
    labels = torch.Tensor(class_numbers)
    labels = labels.type(torch.LongTensor)
    labels = labels.to(device)

    fake_1 = fake_1.view([fake_1.shape[0],-1])
    fake_2 = fake_2.view([fake_1.shape[0],-1])
    real = real.view([real.shape[0],-1])

    real_norm = torch.norm(real,2,dim = 1,keepdim = True)
    fake_norm_1 = torch.norm(fake_1,2,dim = 1,keepdim = True)
    fake_norm_2 = torch.norm(fake_2,2,dim = 1,keepdim = True)

    scores = torch.mm(real.detach(),fake_1.transpose(0,1))
    norm = torch.mm(real_norm.detach(),fake_norm_1.transpose(0,1))
    scores_1 = scores/norm.clamp(min=1e-8)

    scores = torch.mm(real.detach(),fake_2.transpose(0,1))
    norm = torch.mm(real_norm.detach(),fake_norm_2.transpose(0,1))
    scores_2 = scores/norm.clamp(min=1e-8)

    # scores = torch.mm(fake_1,fake_2.transpose(0,1))
    # norm = torch.mm(fake_norm_1,fake_norm_2.transpose(0,1))
    # scores_3 = scores/norm.clamp(min=1e-8)

    value = (1/2)*(torch.mean(criterion_cross(scores_2,labels))+
                    torch.mean(criterion_cross(scores_1,labels)))

    return value

def contrastive_loss_images(fake,real):

    fake = fake.mean(-1).mean(-1)
    real = real.mean(-1).mean(-1)

    class_numbers = np.arange(0,real.shape[0])
    labels = torch.Tensor(class_numbers)
    labels = labels.type(torch.LongTensor)
    labels = labels.to(device)

    fake = fake.view([fake.shape[0],-1])
    real = real.view([real.shape[0],-1])

    real_norm = torch.norm(real,2,dim = 1,keepdim = True)
    fake_norm = torch.norm(fake,2,dim = 1,keepdim = True)

    scores = torch.mm(real.detach(),fake.transpose(0,1))
    norm = torch.mm(real_norm.detach(),fake_norm.transpose(0,1))

    scores = scores/norm.clamp(min=1e-8)

    value = torch.mean(criterion_cross(scores,labels)) + torch.mean(criterion_cross(scores.transpose(0, 1),labels))

    return value


def contrastive_loss_images_word(fake,real):


    class_numbers = np.arange(0,real.shape[0])
    labels = torch.Tensor(class_numbers)
    labels = labels.type(torch.LongTensor)
    labels = labels.to(device)

    fake = fake.view([fake.shape[0],-1])
    real = real.view([real.shape[0],-1])

    real_norm = torch.norm(real,2,dim = 1,keepdim = True)
    fake_norm = torch.norm(fake,2,dim = 1,keepdim = True)

    scores = torch.mm(real.detach(),fake.transpose(0,1))
    norm = torch.mm(real_norm.detach(),fake_norm.transpose(0,1))

    scores = scores/norm.clamp(min=1e-8)

    value = torch.mean(criterion_cross(scores,labels)) + torch.mean(criterion_cross(scores.transpose(0, 1),labels))

    return value

def return_seg(fake):
    
    a,_ = fake.max(dim = -1)
    a,_ = a.max(dim = -1)
    a = a.view([-1,1,1,1])
    fake = fake / a
    fake = F.interpolate(fake,(64,64),mode='bilinear')
    # print(fake.shape)

    return fake

def save_img(images,path):

    z = np.transpose(vutils.make_grid(images.to(device),padding=2, normalize=True,nrow = 4).cpu(),(1,2,0))
    z = z.numpy()
    matplotlib.image.imsave(path+'.jpg', z)

    return 0

def save_map(images,path):

    # z = np.transpose(vutils.make_grid(fake_1_c.to(device)[:36],padding=2, normalize=True,nrow = 5).cpu(),(1,2,0))
    # z = z.numpy()
    # matplotlib.image.imsave(path_main_attn_64+'.jpg', z,cmap = 'jet'

    # print(images.shape)
    z = np.transpose(vutils.make_grid(images.to(device),padding=2, normalize=True,nrow = 4).cpu(),(1,2,0))
    # print(z.shape)
    z = z.numpy()
    z = z[:,:,0]
    matplotlib.image.imsave(path+'.jpg', z, cmap = 'gray')

    return 0

def multi_acc(tags, label):
    corrects = (tags == label).float()
    acc = corrects.sum() / corrects.numel()
    acc = acc * 100
    return acc

class FocalLoss(nn.Module):
    """Implementation of Facal Loss"""
    def __init__(self, weight=None, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.weighted_cs = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.cs = nn.CrossEntropyLoss(reduction="none")
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, predicted, target):
        """
        predicted: [batch_size, n_classes]
        target: [batch_size]
        """
        pt = 1/torch.exp(self.cs(predicted,target))
        #shape: [batch_size]
        entropy_loss = self.weighted_cs(predicted, target)
        #shape: [batch_size]
        focal_loss = ((1-pt)**self.gamma)*entropy_loss
        #shape: [batch_size]
        if self.reduction =="none":
            return focal_loss
        elif self.reduction == "mean":
            return focal_loss.mean()
        else:
            return focal_loss.sum()