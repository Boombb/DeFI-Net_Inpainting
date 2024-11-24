import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Swish


class TopDown_Fusion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(TopDown_Fusion, self).__init__()
        
        self.conv1x1_dict = nn.ModuleDict({
            '101':nn.Conv2d(in_channels=in_channel[0], out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            '102':nn.Conv2d(in_channels=in_channel[1], out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            '103':nn.Conv2d(in_channels=in_channel[2], out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            '104':nn.Conv2d(in_channels=in_channel[3], out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            '1':nn.Conv2d(in_channels=out_channel*2, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            '2':nn.Conv2d(in_channels=out_channel*4, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            '3':nn.Conv2d(in_channels=out_channel*4, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            '4':nn.Conv2d(in_channels=out_channel*3, out_channels=out_channel, kernel_size=1, stride=1, padding=0),

        })
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channel),
                Swish(inplace=True)
            ) for _ in range(4)
        ])
        
        self.down_sample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    
    def forward(self, x):
        
        # input out= [x1, x2, x3, x4]
        x1, x2, x3, x4= x
        
        # channel_reduce --> Eq(1), j=1
        x1 = self.conv1x1_dict['101'](x1)
        x2 = self.conv1x1_dict['102'](x2)
        x3 = self.conv1x1_dict['103'](x3)
        x4 = self.conv1x1_dict['104'](x4)

        # down_sample
        x1_d = self.down_sample(x1)
        x2_d = self.down_sample(x2)
        x3_d = self.down_sample(x3)
  
        
        # up_sample
        x2_u = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x3_u = F.interpolate(x3, scale_factor=2, mode='bilinear')
        x4_u = F.interpolate(x4, scale_factor=2, mode='bilinear')

        
        # fusion --> Eq(1), j=2
        # x4
        x4_ = torch.cat((x3_d, x4), dim=1)
        x4_ = self.conv1x1_dict['1'](x4_)
        x4_f = self.blocks[0](x4_)
        
        # x3
        x4_u_ = F.interpolate(x4_, scale_factor=2, mode='bilinear')
        x3_ = torch.cat((x2_d, x3, x4_u, x4_u_), dim=1)
        x3_ = self.conv1x1_dict['2'](x3_)
        x3_f = self.blocks[1](x3_)
        
        # x2
        x3_u_ = F.interpolate(x3_, scale_factor=2, mode='bilinear')
        x2_ = torch.cat((x1_d, x2, x3_u, x3_u_), dim=1)
        x2_ = self.conv1x1_dict['3'](x2_)
        x2_f = self.blocks[2](x2_)
        
        # x1
        x2_u_ = F.interpolate(x2_, scale_factor=2, mode='bilinear')
        x1_ = torch.cat((x1, x2_u, x2_u_), dim=1)
        x1_ = self.conv1x1_dict['4'](x1_)
        x1_f = self.blocks[3](x1_)
        
        return [x1_f, x2_f, x3_f, x4_f]
    


class BottomUp_Fusion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BottomUp_Fusion, self).__init__()
        
        self.conv1x1_dict = nn.ModuleDict({
            '1':nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            '2':nn.Conv2d(in_channels=in_channel*4, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            '3':nn.Conv2d(in_channels=in_channel*4, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            '4':nn.Conv2d(in_channels=in_channel*3, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
        })
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channel),
                Swish(inplace=True)
            ) for _ in range(4)
        ])
        
        self.down_sample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    
    def forward(self, x):
        
        # input x= [x1, x2, x3, x4]
        x1, x2, x3, x4 = x
                
        # down_sample
        x1_d = self.down_sample(x1)
        x2_d = self.down_sample(x2)
        x3_d = self.down_sample(x3)

        
        # up_sample
        x2_u = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x3_u = F.interpolate(x3, scale_factor=2, mode='bilinear')
        x4_u = F.interpolate(x4, scale_factor=2, mode='bilinear')
        
        # fusion --> Eq(1), j=3
        # x1
        x1_ = torch.cat((x1, x2_u), dim=1)
        x1_ = self.conv1x1_dict['1'](x1_)
        x1_f = self.blocks[0](x1_)
        
        # x2
        x1_d_ = self.down_sample(x1_)
        x2_ = torch.cat((x1_d_, x1_d, x2, x3_u), dim=1)
        x2_ = self.conv1x1_dict['2'](x2_)
        x2_f = self.blocks[1](x2_)
        
        # x3
        x2_d_ = self.down_sample(x2_)
        x3_ = torch.cat((x2_d_, x2_d, x3, x4_u), dim=1)
        x3_ = self.conv1x1_dict['3'](x3_)
        x3_f = self.blocks[2](x3_)
        
        # x4
        x3_d_ = self.down_sample(x3_)
        x4_ = torch.cat((x3_d_, x3_d, x4), dim=1)
        x4_ = self.conv1x1_dict['4'](x4_)
        x4_f = self.blocks[3](x4_)
    
        return [x1_f, x2_f, x3_f, x4_f]
    
    
class DFPL(nn.Module):
    def __init__(self, fpn_in_channel, fpn_out_channel):
        super(DFPL, self).__init__()
        
        self.fpn_in_channel = fpn_in_channel
        self.fpn_out_channel = fpn_out_channel
        
        
        self.td_fusion = TopDown_Fusion(in_channel=self.fpn_in_channel, out_channel=self.fpn_out_channel)
        self.bu_fusion = BottomUp_Fusion(in_channel=self.fpn_out_channel, out_channel=self.fpn_out_channel)
        
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=self.fpn_in_channel[i]+self.fpn_out_channel, out_channels=self.fpn_out_channel, kernel_size=1, stride=1, padding=0),
                nn.Conv2d(in_channels=self.fpn_out_channel, out_channels=self.fpn_out_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.fpn_out_channel),
                Swish(inplace=True)
            ) for i in range(4)
        ])
    
    def forward(self, x):
        
        # input x= [x1, x2, x3, x4]
        x1, x2, x3, x4 = x 
        
        # TopDown_Fusion out_x = [x1, x2, x3, x4]
        x = self.td_fusion(x)
        
        # BottomUp_Fusion out_x = [x1, x2, x3, x4]
        x = self.bu_fusion(x)
        
        x1_, x2_, x3_, x4_ = x
        
        # x1
        x1_f = torch.cat((x1, x1_), dim=1)
        x1_f = self.blocks[0](x1_f)
        
        # x2
        x2_f = torch.cat((x2, x2_), dim=1)
        x2_f = self.blocks[1](x2_f)
        
        # x3
        x3_f = torch.cat((x3, x3_), dim=1)
        x3_f = self.blocks[2](x3_f)
        
        # x4
        x4_f = torch.cat((x4, x4_), dim=1)
        x4_f = self.blocks[3](x4_f)
        
        
        return [x1_f, x2_f, x3_f, x4_f]

