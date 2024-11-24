import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from model.Network import Network
from data.dataset import test_dataset
from utils.IoU import IoU
from sklearn.metrics import roc_auc_score, f1_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./pre_train/Net_epoch_best_IIDdata.pth')
parser.add_argument('--data_path', type=str, default='./dataset/IIDdata/test/')
parser.add_argument('--resSave_path', type=str, default='./output/IID')
opt = parser.parse_args()

if not os.path.exists(opt.resSave_path):
    os.makedirs(opt.resSave_path)

model = Network()

pytorch_total_params = sum(p.numel() for p in Network().parameters() if p.requires_grad)
print('Total Params: %d' % pytorch_total_params)

model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(opt.pth_path))
model.cuda()
model.eval()

image_root = '{}/images/'.format(opt.data_path)
gt_root = '{}/labels/'.format(opt.data_path)
test_loader = test_dataset(image_root, gt_root, opt.testsize, dataset_name='IID')
    
    
total_IoU = 0
total_AUC = 0
total_F1 = 0
for i in range(test_loader.size):
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    gt[gt>=0.5] = 1
    gt[gt<0.5] = 0
    H, W = gt.shape
    image = image.cuda()
    
    edge, updated_res = model(image)
        
    updated_res = F.interpolate(updated_res, size=gt.shape, mode='bilinear', align_corners=False)
    updated_res = updated_res.sigmoid().data.cpu().numpy().squeeze()
    updated_res = (updated_res - updated_res.min()) / (updated_res.max() - updated_res.min() + 1e-8)
    
    edge = F.interpolate(edge, size=gt.shape, mode='bilinear', align_corners=False)
    edge = edge.sigmoid().data.cpu().numpy().squeeze()
    edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
    edge[edge>=0.5] = 1
    edge[edge<0.5] = 0

    try:
        total_AUC += roc_auc_score(gt.reshape(H * W).astype('int'), updated_res.reshape(H * W)) * 100. 
    except:
        pass
    
    updated_res[updated_res>=0.5] = 1
    updated_res[updated_res<0.5] = 0
    
    metric = IoU(updated_res.reshape(H * W), gt.reshape(H * W))
    total_IoU += metric.get_IoU() * 100.
    total_F1 += f1_score(gt.reshape(H * W).astype('int'), updated_res.reshape(H * W).astype('int'), average='macro') * 100.

    # saveRes
    save_img = np.hstack((updated_res*255, edge*255, gt*255))
    
    
    print('>{}'.format(name))
    save_name = name.split('.')[0] + '.png'
    cv2.imwrite(os.path.join(opt.resSave_path, save_name), save_img)

    
f1 = total_F1 / test_loader.size
iou = total_IoU / test_loader.size
auc = total_AUC / test_loader.size
print('F1:{}, IoU: {}, AUC: {}'.format(f1, iou, auc))
print(test_loader.size)