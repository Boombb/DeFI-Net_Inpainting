# author: Daniel-Ji (e-mail: gepengai.ji@gmail.com)
# data: 2021-01-16
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import torch.backends.cudnn as cudnn

from datetime import datetime
from torchvision.utils import make_grid
from model.Network import Network
from data.dataset import get_loader, test_dataset
from utils.utils import clip_gradient
from utils.IoU import IoU
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.loss import structure_loss, bondary_loss



def train(train_loader, model, optimizer, edge_lossWeigt, seg_lossWeigt, epoch, save_path, writer):
    """
    train function
    """
    global step, total_step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, edge_gts) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            edge_gts = edge_gts.cuda()
            
            edge_preds, seg_preds = model(images)


            loss_seg = structure_loss(seg_preds, gts) * seg_lossWeigt
            loss_edge = bondary_loss(edge_preds, edge_gts) * edge_lossWeigt
            loss = loss_seg + loss_edge
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 1000 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss_edge: {:.4f} Loss_seg: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_edge.data, loss_seg.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss_edge: {:.4f} '
                    'Loss_seg: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_edge.data, loss_seg.data))

                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_edge': loss_edge.data, 'Loss_seg': loss_seg.data,
                                    'Loss_total': loss.data},
                                   global_step=step)
                
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)
                grid_image = make_grid(edge_gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Edge_GT', grid_image, step)

                # TensorboardX-Outputs
                res = edge_preds[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_edge', torch.tensor(res), step, dataformats='HW')
                res = seg_preds[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_seg', torch.tensor(res), step, dataformats='HW')


        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 50 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_auc, best_epoch
    model.eval()
    with torch.no_grad():
        total_F1 = 0
        total_IoU = 0
        total_AUC = 0
        for i in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            gt[gt>=0.5] = 1
            gt[gt<0.5] = 0
            H, W = gt.shape
            image = image.cuda()

            _, pred_res = model(image)
            pred_res = F.interpolate(pred_res, size=gt.shape, mode='bilinear', align_corners=False)
            pred_res = pred_res.sigmoid().data.cpu().numpy().squeeze()
            pred_res = (pred_res - pred_res.min()) / (pred_res.max() -  pred_res.min() + 1e-8)
            
            # metric
            try:
                total_AUC += roc_auc_score(gt.reshape(H * W).astype('int'), pred_res.reshape(H * W))
            except:
                pass
            pred_res[pred_res>=0.5] = 1
            pred_res[pred_res<0.5] = 0
            total_F1 += f1_score(gt.reshape(H * W).astype('int'), pred_res.reshape(H * W).astype('int'), average='macro')
            metric = IoU(pred_res, gt)
            total_IoU += metric.get_IoU()    
        
        f1 = total_F1 / test_loader.size
        iou = total_IoU / test_loader.size
        auc = total_AUC / test_loader.size
        writer.add_scalar('F1', torch.tensor(f1), global_step=epoch)
        writer.add_scalar('AUC', torch.tensor(auc), global_step=epoch)
        writer.add_scalar('IOU', torch.tensor(iou), global_step=epoch)
        print('Epoch: {}, F1: {}, IoU: {}, AUC: {}, bestAUC: {}, bestEpoch: {}.'.format(epoch, f1, iou, auc, best_auc, best_epoch))
        if epoch == 1:
            best_auc = auc
            torch.save(model.state_dict(), save_path + 'Net_epoch_best_{:0.5f}.pth'.format(best_auc))
            print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        else:
            if best_auc < auc:
                best_auc = auc
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best_{:0.5f}.pth'.format(best_auc))
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} F1:{} IoU:{} AUC:{} bestEpoch:{} bestAUC:{}'.format(epoch, f1, iou, auc, best_epoch, best_auc))
        
    return auc


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--edgeWeight', type=float, default=1, help='edge loss weight')
    parser.add_argument('--segWeight', type=float, default=5, help='seg loss weight')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='-1', help='train use gpu')
    parser.add_argument('--dataset_name', type=str, default='IID', help='name of dataset')
    parser.add_argument('--train_root', type=str, default='./dataset/IIDdata/train/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='./dataset/IIDdata/test/',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./save/IID_train',
                        help='the path to save model and log')
    opt = parser.parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
        print('USE ALL GPU')
    cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # build the model
    model = Network()
    model = nn.DataParallel(model)
    model.to(device)
    
    pytorch_total_params = sum(p.numel() for p in Network().parameters() if p.requires_grad)
    print('Total Params: %d' % pytorch_total_params)

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'images/',
                              gt_root=opt.train_root + 'labels/',
                              edge_root=opt.train_root + 'edges/',
                              dataset_name=opt.dataset_name,
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=12)
    val_loader = test_dataset(image_root=opt.val_root + 'images/',
                              gt_root=opt.val_root + 'labels/',
                              testsize=opt.trainsize, dataset_name=opt.dataset_name)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; load: {}; '
                 'save_path: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.load, save_path))

    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_auc = 0
    best_epoch = 0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, opt.edgeWeight, opt.segWeight, epoch, save_path, writer)
        val_auc = val(val_loader, model, epoch, save_path, writer)
        scheduler.step(val_auc)
