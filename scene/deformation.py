import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import torch.nn.init as init
from scene.hexplane import HexPlaneField


class Linear_Res(nn.Module):
    def __init__(self, W):
        super(Linear_Res, self).__init__()
        self.main_stream = nn.Linear(W, W)

    def forward(self, x):
        x = F.relu(x)
        return x + self.main_stream(x)


class Feat_Res_Net(nn.Module):
    def __init__(self, W, D):
        super(Feat_Res_Net, self).__init__()
        self.D = D
        self.W = W
    
        self.feature_out = [Linear_Res(self.W)]
        for i in range(self.D-2):
            self.feature_out.append(Linear_Res(self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
    
    def initialize_weights(self,):
        for m in self.feature_out.modules():
            if isinstance(m, nn.Linear):
                init.constant_(m.weight, 0)
                # init.xavier_uniform_(m.weight,gain=1)
                if m.bias is not None:
                    # init.xavier_uniform_(m.bias,gain=1)
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.feature_out(x)


class Head_Res_Net(nn.Module):
    def __init__(self, W, H):
        super(Head_Res_Net, self).__init__()
        self.W = W
        self.H = H

        self.feature_out = [Linear_Res(self.W)]
        self.feature_out.append(nn.Linear(W, self.H))
        self.feature_out = nn.Sequential(*self.feature_out)
    
    def initialize_weights(self,):
        for m in self.feature_out.modules():
            if isinstance(m, nn.Linear):
                init.constant_(m.weight, 0)
                # init.xavier_uniform_(m.weight,gain=1)
                if m.bias is not None:
                    # init.xavier_uniform_(m.bias,gain=1)
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.feature_out(x)



class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, skips=[], args=None, use_res=False):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips

        self.no_grid = args.no_grid # False
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)

        self.use_res = use_res
        if not self.use_res:
            self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = self.create_net()
        else:
            self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = self.create_res_net()
        self.args = args
    
    def create_net(self):
        
        mlp_out_dim = 0
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        output_dim = self.W
        return  \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
    
    def create_res_net(self,):
        
        mlp_out_dim = 0

        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + self.grid.feat_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)

        # self.feature_in = nn.Linear(mlp_out_dim + self.grid.feat_dim ,self.W)
        # self.feature_out = Feat_Res_Net(self.W, self.D)

        output_dim = self.W
        return  \
            Head_Res_Net(self.W, 3), \
            Head_Res_Net(self.W, 3), \
            Head_Res_Net(self.W, 4), \
            Head_Res_Net(self.W, 1) 

    
    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_emb):

        if not self.use_res:
            if self.no_grid:
                h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
            else:
                grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])

                h = grid_feature
            
            h = self.feature_out(h)
        else:
            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            # h =  self.feature_out(self.feature_in(grid_feature))
            h = self.feature_out(grid_feature)
        return h

    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity = None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, opacity, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx

    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, opacity_emb, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_emb).float()
        dx = self.pos_deform(hidden)
        pts = rays_pts_emb[:, :3] + dx
        if self.args.no_ds: # False
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)
            scales = scales_emb[:,:3] + ds
        if self.args.no_dr: # False
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)
            rotations = rotations_emb[:,:4] + dr
        if self.args.no_do: # True
            opacity = opacity_emb[:,:1] 
        else:
            do = self.opacity_deform(hidden) 
            opacity = opacity_emb[:,:1] + do
        # + do
        # print("deformation value:","pts:",torch.abs(dx).mean(),"rotation:",torch.abs(dr).mean())

        return pts, scales, rotations, opacity
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    def get_grid_parameters(self):
        return list(self.grid.parameters() ) 
    # + list(self.timegrid.parameters())


class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        
        self.use_res = args.use_res
        if self.use_res:
            print("Using zero-init and residual")
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(4+3)+((4+3)*scale_rotation_pe)*2, input_ch_time=timenet_output, args=args, use_res=self.use_res)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        self.apply(initialize_weights)

        if self.use_res:
            # self.deformation_net.feature_out.initialize_weights()
            self.deformation_net.pos_deform.initialize_weights()
            self.deformation_net.scales_deform.initialize_weights()
            self.deformation_net.rotations_deform.initialize_weights()
            self.deformation_net.opacity_deform.initialize_weights()

        # self.deformation_net.feature_out.apply(initialize_zeros_weights)
        # print(self)

    def forward(self, point, scales=None, rotations=None, opacity=None, times_sel=None):
        if times_sel is not None:
            return self.forward_dynamic(point, scales, rotations, opacity, times_sel)
        else:
            return self.forward_static(point)

        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    def forward_dynamic(self, point, scales=None, rotations=None, opacity=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)

        means3D, scales, rotations, opacity = self.deformation_net( point,
                                                  scales,
                                                rotations,
                                                opacity,
                                                # times_feature,
                                                times_sel)
        return means3D, scales, rotations, opacity
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.xavier_uniform_(m.bias,gain=1)
            # init.constant_(m.bias, 0)

def initialize_zeros_weights(m):
    if isinstance(m, nn.Linear):
        init.constant_(m.weight, 0)
        # init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            # init.xavier_uniform_(m.bias,gain=1)
            init.constant_(m.bias, 0)
