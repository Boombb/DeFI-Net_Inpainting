import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.hr_config import get_hrnet_cfg
from .backbone.hrnet import get_seg_model
from .DFPL import DFPL




class Network(nn.Module):
    def __init__(self, dfpl_in_channel=[18, 36, 72, 144], dfpl_out_channel=64):
        super(Network, self).__init__()
        # ---- Backbone ----
        self.hr_config = get_hrnet_cfg()
        self.hrnet = get_seg_model(self.hr_config)
        
        # ---- Dense Feature Pyramid Learning ----
        self.gfpn = DFPL(fpn_in_channel=dfpl_in_channel, fpn_out_channel=dfpl_out_channel)
        
        # ---- Reverse Edge Attention Enhancement ----
        self.ream = REAM(low1_in_channel=dfpl_out_channel, low2_in_channel=dfpl_out_channel, mid_in_channel=dfpl_out_channel, out_channel=dfpl_out_channel)
        
        # ---- Spatial Adaptive Feature Fusion ----
        self.dff = SAFF(in_channels=dfpl_out_channel)


    def forward(self, x):
        
        # Feature Extraction
        x1, x2, x3, x4 = self.hrnet(x)
        
        # Dense Feature Pyramid Learning
        x_fusion = [x1, x2, x3, x4]
        x_fusion = self.gfpn(x_fusion)
        x_low1, x_low2, x_mid, x_high = x_fusion
        
        # Reverse Edge Attention Enhancement
        low_mid_fuse, edge_pred = self.ream(x_low1, x_low2, x_mid)
        
        # Spatial Adaptive Feature Fusion
        high_seg_pred = self.dff(low_mid_fuse, x_high)
        
        return edge_pred, high_seg_pred
        

class SAFF(nn.Module):
    def __init__(self , in_channels , nclass=2, layernum=2, norm_layer=nn.BatchNorm2d):
        super(SAFF, self).__init__()
        
        self.nclass = nclass
        
        self.ada_learner = LocationAdaptiveLearner(nclass, nclass*layernum, nclass*layernum, norm_layer=norm_layer)
        
        self.low_mid = nn.Sequential(nn.Conv2d(in_channels, nclass, 1, bias=True),
                                   norm_layer(nclass))
        
        self.high = nn.Sequential(nn.Conv2d(in_channels, nclass, 1, bias=True),
                                   norm_layer(nclass),
                                   nn.ConvTranspose2d(nclass, nclass, 16, stride=8, padding=4, bias=False))
        
        self.high_prior = nn.Sequential(nn.Conv2d(in_channels, nclass*layernum, 1, bias=True),
                                   norm_layer(nclass*layernum),
                                   nn.ConvTranspose2d(nclass*layernum, nclass*layernum, 16, stride=8, padding=4, bias=False))
        
        self.seghead = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=1)
        
        
    def forward(self, low_mid, high):
        high_w = self.high_prior(high) 

        low_mid = self.low_mid(low_mid) # (N, 2, H, W)
        high = self.high(high) # (N, 2, H, W)
        
        weights = self.ada_learner(high_w) # (N, 2, 2, H, W) 

    
        slicehigh_0 = high[:,0:1,:,:] # (N, 1, H, W)
        slice_lowmid_0 = low_mid [:,0:1,:,:] # (N, 1, H, W)
        fuse = torch.cat((slicehigh_0, slice_lowmid_0), dim=1)
        slicehigh_1 = high[:,1:2,:,:] # (N, 1, H, W)
        slice_lowmid_1 = low_mid [:,1:2,:,:] # (N, 1, H, W)
        fuse = torch.cat((fuse, slicehigh_1, slice_lowmid_1), dim=1) # (N, 2*2, H, W)

        fuse = fuse.view(fuse.size(0), self.nclass, -1, fuse.size(2), fuse.size(3)) # (N, 2, 2, H, W)
        fuse = torch.mul(fuse, weights) # (N, 2, 2, H, W)
        fuse = torch.sum(fuse, 2) # (N, 2, H, W)
        fuse = self.seghead(fuse)
        return fuse
    

class LocationAdaptiveLearner(nn.Module):
    def __init__(self, nclass, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(LocationAdaptiveLearner, self).__init__()
        self.nclass = nclass

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels))

    def forward(self, x):
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.conv3(x)
        x = x.view(x.size(0), self.nclass, -1, x.size(2), x.size(3))
        return x

    
class REAM(nn.Module):
    def __init__(self, low1_in_channel, low2_in_channel, mid_in_channel, out_channel):
        super(REAM, self).__init__()

        self.edge_head = segmenthead(inplanes=low1_in_channel, interplanes=out_channel, outplanes=1, scale_factor=1)
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        
        # fusion
        self.conv3x3_mid = nn.Conv2d(in_channels=mid_in_channel, out_channels=mid_in_channel, kernel_size=3, stride=1,padding=1)
        self.bn_mid = nn.BatchNorm2d(mid_in_channel)
        self.conv3x3_low2 = nn.Conv2d(in_channels=low2_in_channel, out_channels=low2_in_channel, kernel_size=3, stride=1, padding=1)
        self.bn_low2 = nn.BatchNorm2d(low2_in_channel)
        self.conv3x3_low1 = nn.Conv2d(in_channels=low1_in_channel, out_channels=low1_in_channel, kernel_size=3, stride=1, padding=1)
        self.bn_low1 = nn.BatchNorm2d(low1_in_channel)
        self.conv3x3_fuse = nn.Conv2d(in_channels=low1_in_channel, out_channels=low1_in_channel, kernel_size=3, stride=1, padding=1)
        self.bn_fuse = nn.BatchNorm2d(low1_in_channel)
        self.relu = nn.ReLU()

    def forward(self, low_1, low_2, mid):
        low_1_h, low_1_w = low_1.shape[2:]
        low_2_h, low_2_w = low_2.shape[2:]
        
        # fuse mid low2
        mid_ = self.conv3x3_mid(mid)
        mid_ = self.bn_mid(mid_)
        mid_up = F.interpolate(mid_, size=(low_2_h, low_2_w), mode='bilinear', align_corners=False)
        low_2_ = self.conv3x3_low2(low_2)
        low_2_ = self.bn_low2(low_2_)
        feat_fuse = self.relu(mid_up + low_2_)
        
        
        #fuse mid low2 low1
        feat_fuse_ = self.conv3x3_fuse(feat_fuse)
        feat_fuse_ = self.bn_fuse(feat_fuse_)
        feat_fuse_up = F.interpolate(feat_fuse_, size=(low_1_h, low_1_w), mode='bilinear', align_corners=False)
        low_1_ = self.conv3x3_low1(low_1)
        low_1_ = self.bn_low1(low_1_)
        feat_fuse = self.relu(feat_fuse_up + low_1_)
        
        avg_mid = torch.mean(mid, dim=1, keepdim=True)
        max_mid, _ = torch.max(mid, dim=1, keepdim=True)

        x = torch.cat([avg_mid, max_mid], dim=1)

        x = 1 - self.conv1(x).sigmoid()
        
        x = F.interpolate(x, size=(low_1_h, low_1_w), mode='bilinear', align_corners=False)
        
        fuse = feat_fuse * x
        
        _, edge_pred = self.edge_head(fuse)
        
        return feat_fuse, edge_pred



class segmenthead(nn.Module):

    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None, bn_mom = 0.1):
        super(segmenthead, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))

        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                        size=[height, width],
                        mode='bilinear', align_corners=False)       
            
        return x, out
