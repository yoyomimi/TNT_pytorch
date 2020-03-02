# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Created by Mingfei Chen (lasiafly@gmail.com)
# Created On: 2020-2-24
# ------------------------------------------------------------------------------
import torch
from torch import nn

from units.units import tracklet_conv2d_with_maxpool


class TrackletConnectivity(nn.Module):
    def __init__(self, cfg, drop_prob=0.25):

        super().__init__()
        assert len(cfg.MODEL.CONNECTIVITY.FEAT_CHANNELS) == cfg.MODEL.CONNECTIVITY.STACK_NUM

        self.ks = cfg.MODEL.CONNECTIVITY.KERNEL_SIZE
        self.channels = cfg.MODEL.CONNECTIVITY.FEAT_CHANNELS
        self.stack_num = cfg.MODEL.CONNECTIVITY.STACK_NUM
        self.window_len = cfg.TRACKLET.WINDOW_lEN
         
        self.conv1 = nn.ModuleList([])
        for i in range(len(self.ks)):
            self.conv1.append(tracklet_conv2d_with_maxpool(3, self.channels[0], ks_kernel_size=(1,self.ks[i]), 
                pool_stride=(1,2), ks_stride=(1,1)))
        
        self.conv2 = nn.ModuleList([])
        for i in range(len(self.ks)):
            self.conv2.append(tracklet_conv2d_with_maxpool(self.channels[0]*4, self.channels[1], ks_kernel_size=(1,self.ks[i]), 
                pool_stride=(1,2), ks_stride=(1,1)))

        self.conv3 = nn.ModuleList([])
        for i in range(len(self.ks)):
            self.conv3.append(tracklet_conv2d_with_maxpool(self.channels[1]*4, self.channels[2], ks_kernel_size=(1,self.ks[i]), 
                pool_stride=(1,2), ks_stride=(1,1)))

        self.fc1 = nn.Sequential(
            nn.Linear(int(5*len(self.ks)*self.window_len/2**self.stack_num*self.channels[-1]), 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_prob)
        )

        self.pred_fc = nn.Linear(1024, 2) # [connected conf, not connected conf]
        
        self.softmax = nn.Softmax(dim=1)

        # TODO
        # self.init_weights()


    def forward(self, x):
        """
        Args:
            x: (bs, window_len, emb_dim+4, 3)
        
        Return:
            similarity: (bs, 1) ranging from 0 to 1
        """
        # (bs, emb_dim+4, T, 3)
        x = x.permute(0, 3, 2, 1)
        bs = x.shape[0]

        # layer_1 -> (bs, emb_dim+4, T/2, 4*16)
        x_temp = []
        for conv in self.conv1:
            x_temp.append(conv(x))
        x = torch.cat(x_temp, 1)
        
        # layer_2 -> (bs, emb_dim+4, T/4, 4*64)
        x_temp = []
        for conv in self.conv2:
            x_temp.append(conv(x))
        x = torch.cat(x_temp, 1)

        # layer_3 with avg on appearance -> (bs, 4*(4+1)*T/8*128)
        x_temp = []
        for conv in self.conv3:
            x_out_temp = conv(x)
            x_loc_temp = x_out_temp[:,:,:4,:]
            x_ap_temp = torch.mean(x_out_temp[:,:,4:,:], dim=2).unsqueeze(2)
            x_out_temp = torch.cat([x_loc_temp, x_ap_temp], dim=2)
            x_temp.append(x_out_temp.reshape([bs, 5*x_out_temp.shape[-1]*x_out_temp.shape[1]]))
        x = torch.cat(x_temp, -1)

        # fc_1 -> (bs, 2048)
        x = self.fc1(x)
        # fc_2 -> (bs, 1024)
        x = self.fc2(x)
        # fc_clssfier -> (bs, 1)
        out = self.softmax(self.pred_fc(x))

        return out


if __name__ == "__main__":
    from configs import cfg
    from torch.autograd import Variable

    model = TrackletConnectivity()
    x = Variable(torch.randn(16, 64, 516, 3))

    print(model(x))


       
            