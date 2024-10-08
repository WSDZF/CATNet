
import torch
import torch.nn as nn
import einops
from .main_blocks import *
from .pca_utils import *



class ChannelAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.q_map = depthwise_projection(in_features=out_features, 
                                            out_features=out_features, 
                                            groups=out_features)
        self.k_map = depthwise_projection(in_features=in_features, 
                                            out_features=in_features, 
                                            groups=in_features)
        self.v_map = depthwise_projection(in_features=in_features, 
                                            out_features=in_features, 
                                            groups=in_features) 

        self.projection = depthwise_projection(in_features=out_features, 
                                    out_features=out_features, 
                                    groups=out_features)
        self.sdp = ScaleDotProduct()        
        

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)
        b, hw, c_q = q.shape
        c = k.shape[2]
        scale = c ** -0.5                     
        q = q.reshape(b, hw, self.n_heads, c_q // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        v = v.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3).transpose(2, 3)
        att = self.sdp(q, k ,v, scale).permute(0, 3, 1, 2).flatten(2)
        att = self.projection(att)
        return att

class SpatialAttention(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4) -> None:
        super().__init__()
        self.n_heads = n_heads

        self.q_map = depthwise_projection(in_features=in_features, 
                                            out_features=in_features, 
                                            groups=in_features)
        self.k_map = depthwise_projection(in_features=in_features, 
                                            out_features=in_features, 
                                            groups=in_features)
        self.v_map = depthwise_projection(in_features=out_features, 
                                            out_features=out_features, 
                                            groups=out_features)       

        self.projection = depthwise_projection(in_features=out_features, 
                                    out_features=out_features, 
                                    groups=out_features)                                             
        self.sdp = ScaleDotProduct()        

    def forward(self, x):
        q, k, v = x[0], x[1], x[2]
        q = self.q_map(q)
        k = self.k_map(k)
        v = self.v_map(v)  
        b, hw, c = q.shape
        c_v = v.shape[2]
        scale = (c // self.n_heads) ** -0.5        
        q = q.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(b, hw, self.n_heads, c_v // self.n_heads).permute(0, 2, 1, 3)
        att = self.sdp(q, k ,v, scale).transpose(1, 2).flatten(2)    
        x = self.projection(att)
        return x


class CABlock(nn.Module):
    def __init__(self,
                 features,
                 channel_head) -> None:
        super().__init__()
        self.channel_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                        eps=1e-6)
                                           for in_features in features])

        self.c_attention = nn.ModuleList([ChannelAttention(
            in_features=sum(features),
            out_features=feature,
            n_heads=head,
        ) for feature, head in zip(features, channel_head)])


    def forward(self, x):
        x_ca = self.channel_attention(x)
        x = self.m_sum(x, x_ca)
        return x

    def channel_attention(self, x):
        x_c = self.m_apply(x, self.channel_norm)
        x_cin = self.cat(*x_c)
        x_in = [[q, x_cin, x_cin] for q in x_c]
        x_att = self.m_apply(x_in, self.c_attention)
        return x_att

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]

    def cat(self, *args):
        return torch.cat((args), dim=2)


class SABlock(nn.Module):
    def __init__(self,
                 features,
                 spatial_head,) -> None:
        super().__init__()
        self.spatial_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                        eps=1e-6)
                                           for in_features in features])

        self.s_attention = nn.ModuleList([SpatialAttention(
            in_features=sum(features),
            out_features=feature,
            n_heads=head,
        )
            for feature, head in zip(features, spatial_head)])

    def forward(self, x):
        x_sa = self.spatial_attention(x)
        x = self.m_sum(x, x_sa)
        return x

    def spatial_attention(self, x):
        x_c = self.m_apply(x, self.spatial_norm)
        x_cin = self.cat(*x_c)
        x_in = [[x_cin, x_cin, v] for v in x_c]
        x_att = self.m_apply(x_in, self.s_attention)
        return x_att

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]

    def cat(self, *args):
        return torch.cat((args), dim=2)
class CCSABlock(nn.Module):
    def __init__(self, 
                features, 
                channel_head, 
                spatial_head, 
                spatial_att=True, 
                channel_att=True) -> None:
        super().__init__()
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        if self.channel_att:
            self.channel_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                    eps=1e-6) 
                                                    for in_features in features])   

            self.c_attention = nn.ModuleList([ChannelAttention(
                                                in_features=sum(features),
                                                out_features=feature,
                                                n_heads=head, 
                                        ) for feature, head in zip(features, channel_head)])
        if self.spatial_att:
            self.spatial_norm = nn.ModuleList([nn.LayerNorm(in_features,
                                                    eps=1e-6) 
                                                    for in_features in features])          
          
            self.s_attention = nn.ModuleList([SpatialAttention(
                                                    in_features=sum(features),
                                                    out_features=feature,
                                                    n_heads=head, 
                                                    ) 
                                                    for feature, head in zip(features, spatial_head)])

    def forward(self, x):
        if self.channel_att:
            x_ca = self.channel_attention(x)
            x = self.m_sum(x, x_ca)
        if self.spatial_att:
            x_sa = self.spatial_attention(x)
            x = self.m_sum(x, x_sa)
        return x


    def channel_attention(self, x):
        x_c = self.m_apply(x, self.channel_norm)
        x_cin = self.cat(*x_c)
        x_in = [[q, x_cin, x_cin] for q in x_c]
        x_att = self.m_apply(x_in, self.c_attention)
        return x_att    

    def spatial_attention(self, x):
        x_c = self.m_apply(x, self.spatial_norm)
        x_cin = self.cat(*x_c)
        x_in = [[x_cin, x_cin, v] for v in x_c]        
        x_att = self.m_apply(x_in, self.s_attention)
        return x_att 
        

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]    

    def cat(self, *args):
        return torch.cat((args), dim=2)



class PCA(nn.Module):
    def __init__(self,
                features,
                strides,
                patch=28,
                channel_att=True,
                spatial_att=True,   
                n=1,              
                channel_head=[1, 1, 1, 1], 
                spatial_head=[4, 4, 4, 4], 
                ):
        super().__init__()
        self.n = n
        self.features = features
        self.spatial_head = spatial_head
        self.channel_head = channel_head
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.patch = patch
        self.patch_avg = nn.ModuleList([PoolEmbedding(
                                                    pooling = nn.AdaptiveAvgPool2d,            
                                                    patch=patch, 
                                                    )
                                                    for _ in features])                
        self.avg_map = nn.ModuleList([depthwise_projection(in_features=feature,
                                                            out_features=feature, 
                                                            kernel_size=(1, 1),
                                                            padding=(0, 0), 
                                                            groups=feature
                                                            )
                                                    for feature in features])
        if self.spatial_att:
            self.s_attention = nn.ModuleList([
                                            SABlock(features=features,
                                                      spatial_head=spatial_head)
                                                      for _ in range(n)])
        if self.channel_att:
            self.c_attention = nn.ModuleList([
                                            CABlock(features=features,
                                                      channel_head=channel_head)
                                                      for _ in range(n)])
                     
        self.upconvs = nn.ModuleList([UpsampleConv(in_features=feature, 
                                                    out_features=feature,
                                                    kernel_size=(1, 1),
                                                    padding=(0, 0),
                                                    norm_type=None,
                                                    activation=False,
                                                    scale=stride, 
                                                    conv='conv')
                                                    for feature, stride in zip(features, strides)])                                                      
        self.bn_relu = nn.ModuleList([nn.Sequential(
                                                    nn.BatchNorm2d(feature), 
                                                    nn.ReLU()
                                                    ) 
                                                    for feature in features])
        self.conv3 = nn.ModuleList([nn.Sequential(

                                                    nn.Conv2d(feature,feature, kernel_size=3, padding=1, bias= False),
                                                    nn.BatchNorm2d(feature),
                                                    nn.ReLU(),
                                                    nn.Sigmoid()
                                                    )
                                                    for feature in features])

    
    def forward(self, raw):
        x = self.m_apply(raw, self.patch_avg) 
        x = self.m_apply(x, self.avg_map)
        # out1 = x
        for block in self.s_attention:
            x1 = block(x)
        for block in self.c_attention:
            x2 = block(x)
        # out2 = x
        x1 = [self.reshape(i) for i in x1]
        x2 = [self.reshape(i) for i in x2]
        x = self.m_sum(x1, x2)
        x = self.m_apply(x, self.conv3)
        # out3 = x
        x = self.m_apply(x, self.upconvs)
        # out4 = x
        x_out = self.m_sum(x, raw)  
        x_out = self.m_apply(x_out, self.bn_relu)
        # return out1,out2,out3,out4,(*x_out, )      
        return (*x_out, )   

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]  
        
    def reshape(self, x):
        return einops.rearrange(x, 'B (H W) C-> B C H W', H=self.patch) 



if __name__ == '__main__':
    # ras = DCA(n=1, features=[32,64,128,256], strides=[32,16,8,4], patch=11)
    # x1 = torch.randn(1, 32, 352, 352)
    # x2 = torch.randn(1, 64, 176, 176)
    # x3 = torch.randn(1, 128, 88, 88)
    # x4 = torch.randn(1, 256, 44, 44)
    ras = PCA(n=1, features=[256,512,1024,2048], strides=[8,4,2,1], patch=11,spatial_att=False)
    x1 = torch.randn(1, 256, 88, 88)
    x2 = torch.randn(1, 512, 44, 44)
    x3 = torch.randn(1, 1024, 22, 22)
    x4 = torch.randn(1, 2048, 11, 11)
    # ras = DCA(n=1, features=[32,64,128,256], strides=[8,4,2,1], patch=64)
    # x1 = torch.randn(1, 32, 512, 512)
    # x2 = torch.randn(1, 64, 256, 256)
    # x3 = torch.randn(1, 128, 128, 128)
    # x4 = torch.randn(1, 256, 64, 64)

    # out1,out2,out3,out4,(x1,x2,x3,x4) = ras([x1,x2,x3,x4])
    # print(len(out1))
    # print(out1[0].shape)
    # print(out1[1].shape)
    # print(out1[2].shape)
    # print(out1[3].shape)
    # print(out2[0].shape)
    # print(out2[1].shape)
    # print(out2[2].shape)
    # print(out2[3].shape)
    # print(out3[0].shape)
    # print(out3[1].shape)
    # print(out3[2].shape)
    # print(out3[3].shape)
    # print(out4[0].shape)
    # print(out4[1].shape)
    # print(out4[2].shape)
    # print(out4[3].shape)
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)
    print(x4.shape)
