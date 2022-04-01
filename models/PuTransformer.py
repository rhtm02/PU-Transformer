import torch
import torch.nn as nn
import pointops.functions as pointops
from models.util import PixelShuffle1D


class PUTHead(nn.Module):

    def __init__(self,in_channels, out_channels=None):
        super(PUTHead, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels
        self.MLP = nn.Sequential(nn.Conv1d(in_channels,self.out_channels,1,bias=False),
                                 #nn.BatchNorm1d(self.out_channels),
                                 nn.ReLU(inplace=True))

    def forward(self,x):
        '''
        :param x: [B,3,N]
        :return: xyz : [B,3,N],features : [B,out_channels,N]
        '''

        return x,self.MLP(x.permute(0,2,1))


class PositionFusion(nn.Module):
    def __init__(self,in_channels,out_channels,num_neighbors=20):
        super(PositionFusion, self).__init__()

        self.out_channels = in_channels if out_channels is None else out_channels

        self.num_neighbors = num_neighbors + 1

        self.geo_grouper = pointops.QueryAndGroup(nsample=num_neighbors + 1,return_idx=True)
        self.feature_grouper = pointops.QueryAndGroupFeature(nsample=num_neighbors + 1,use_feature=True)

        self.maxpooling = nn.MaxPool2d(kernel_size=(1,num_neighbors + 1))

        self.M_geo = nn.Sequential(nn.Conv2d(6,self.out_channels,1,bias=False),
                                 nn.BatchNorm2d(self.out_channels),
                                 nn.ReLU(inplace=True))

        self.M_feat = nn.Sequential(nn.Conv2d(2 * in_channels,self.out_channels,1,bias=False),
                                 nn.BatchNorm2d(self.out_channels),
                                 nn.ReLU(inplace=True))


    def forward(self,px):
        xyz,f = px

        xyz_diff,grouped_xyz,indices = self.geo_grouper(xyz=xyz) # (B, diff, N, K) (B,3,N,K)
        G_feat = self.feature_grouper(xyz=xyz, features=f, idx=indices.int())  # (B,diff + f, N, K)

        G_geo = xyz.permute(0,2,1).unsqueeze(-1).repeat(1,1,1,self.num_neighbors)
        G_geo = torch.cat([G_geo,xyz_diff],dim=1)

        G_geo = self.M_geo(G_geo)
        G_feat = self.M_feat(G_feat)


        feature = torch.cat([G_geo,G_feat],dim=1)

        return self.maxpooling(feature).squeeze()


def xavier_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight)
    elif classname.find('Linear')!=-1:
        nn.init.xavier_normal(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ShiftedChannelMumtiHead(nn.Module):
    def __init__(self,in_channels, out_channels=None,psi=4):
        super(ShiftedChannelMumtiHead, self).__init__()
        self.out_channels = in_channels if out_channels is None else out_channels

        self.to_q = nn.Sequential(nn.Conv1d(in_channels,self.out_channels,1,bias=False),
                                  nn.BatchNorm1d(self.out_channels),
                                  nn.ReLU(inplace=True)
                                  )
        self.to_k = nn.Sequential(nn.Conv1d(in_channels,self.out_channels,1,bias=False),
                                  nn.BatchNorm1d(self.out_channels),
                                  nn.ReLU(inplace=True)
                                  )
        self.to_v = nn.Sequential(nn.Conv1d(in_channels,self.out_channels,1,bias=False),
                                  nn.BatchNorm1d(self.out_channels),
                                  nn.ReLU(inplace=True)
                                  )

        self.w = self.out_channels // psi
        self.d = self.w // 2
        self.heads = 2 * psi - 1

        self.fusion = nn.Sequential(nn.Conv1d(self.heads * self.w,self.out_channels,1),
                                    nn.BatchNorm1d(self.out_channels),
                                    nn.ReLU(inplace=True))

    def forward(self,x):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        O = []
        i = 0
        for m in range(self.heads):
            i += 1
            q_m = q[:, m * self.d:m * self.d + self.w,:]
            k_m = k[:, m * self.d:m * self.d + self.w,:]
            v_m = v[:, m * self.d:m * self.d + self.w,:]
            a_m = torch.matmul(q_m.permute(0,2,1),k_m)
            a_m = torch.softmax(a_m,dim=-1)
            o_m = torch.matmul(a_m,v_m.permute(0,2,1))
            O.append(o_m.permute(0,2,1))
        O = torch.cat(O,dim=1)
        return self.fusion(O)

class PUTEncoder(nn.Module):
    def __init__(self,in_channels,out_channels,psi=4):
        super(PUTEncoder, self).__init__()
        self.pose_fusion = PositionFusion(in_channels=in_channels,out_channels=out_channels//2,num_neighbors=20)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.sc_mha = ShiftedChannelMumtiHead(in_channels=out_channels, out_channels=out_channels,psi=psi)
        self.layernorm = nn.LayerNorm(out_channels)
        self.mlp = nn.Conv1d(out_channels,out_channels,1)

    def forward(self,xyz_feature):

        xyz,feature = xyz_feature

        feature_ = self.pose_fusion([xyz,feature])
        feature = self.batchnorm(feature_)
        feature = self.sc_mha(feature)

        feature__ = feature + feature_

        feature = self.layernorm(feature__.permute(0,2,1))

        feature = self.mlp(feature.permute(0,2,1))

        feature = feature__ + feature

        return feature

class PUTBody(nn.Module):
    def __init__(self,in_channels=16,out_channels=512,psi=4):
        super(PUTBody, self).__init__()

        self.encoder_1 = PUTEncoder(in_channels=in_channels,out_channels=32,psi=psi)
        self.encoder_2 = PUTEncoder(in_channels=32,out_channels=64, psi=psi)
        self.encoder_3 = PUTEncoder(in_channels=64,out_channels=128, psi=psi)
        self.encoder_4 = PUTEncoder(in_channels=128,out_channels=256, psi=psi)
        self.encoder_5 = PUTEncoder(in_channels=256,out_channels=out_channels, psi=psi)


    def forward(self,xyz_feature):

        xyz,feature = xyz_feature

        feature = self.encoder_1([xyz,feature])
        feature = self.encoder_2([xyz, feature])
        feature = self.encoder_3([xyz, feature])
        feature = self.encoder_4([xyz, feature])
        feature = self.encoder_5([xyz, feature])

        return feature

class PUTTail(nn.Module):
    def __init__(self,in_channels,out_channels,upscale_factor=4):
        super(PUTTail, self).__init__()
        self.shuffle = PixelShuffle1D(upscale_factor=upscale_factor)

        self.mlp = nn.Conv1d(in_channels//upscale_factor,out_channels,1)

    def forward(self,x):

        upsample = self.shuffle(x)

        coordinate = self.mlp(upsample)

        return coordinate

class PointUpsamplingTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super(PointUpsamplingTransformer, self).__init__()

        self.head = PUTHead(in_channels=in_channels, out_channels=16).cuda()
        self.body = PUTBody(in_channels=16, out_channels=512, psi=4).cuda()
        self.tail = PUTTail(in_channels=512, out_channels=out_channels, upscale_factor=upscale_factor).cuda()
    def forward(self,x):
        xyz, feature = self.head(x)
        feature = self.body([xyz, feature])
        coord = self.tail(feature)

        return coord.permute(0,2,1)

if __name__ == '__main__':
    x = torch.randn(64,256,3).cuda()

    head = PUTHead(in_channels=3, out_channels=16).cuda()
    body = PUTBody(in_channels=16,out_channels=512,psi=4).cuda()
    tail = PUTTail(in_channels=512,out_channels=3,upscale_factor=4).cuda()
    xyz,feature = head(x)
    print(xyz.shape,feature.shape)
    feature = body([xyz,feature])
    coord = tail(feature)
    print(xyz.shape,feature.shape,coord.shape)
