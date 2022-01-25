from __future__ import print_function
import argparse
import os
import random
from tkinter import W
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.autograd import Function
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F
import scipy.sparse
import sys


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))  # x = batch,1024,n(n=2048)
        x = torch.max(x, 2, keepdim=True)[0]  # x = batch,1024,1
        x = x.view(-1, 1024)  # x = batch,1024
        if self.global_feat:
            return x, trans
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans


class PointNetCls(nn.Module):
    def __init__(self, k=2):
        super(PointNetCls, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.log_softmax(x, dim=0), trans


# ***************  YW test FoldingNet ************
class FoldingNetEnc(nn.Module):
    def __init__(self):
        super(FoldingNetEnc, self).__init__()
        self.feat = PointNetfeat(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)
        # self.fc2 = nn.Linear(512, 256)
        # self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans = self.feat(x)  # x = batch,1024
        x = F.relu(self.bn1(self.fc1(x)))  # x = batch,512
        x = self.fc2(x)  # x = batch,512

        return x, trans


class Graph_Pooling(nn.Module):
    def __init__(self):
        super(Graph_Pooling,self).__init__()

    def forward(self, x, batch_graph):      # x: batch, channel, num_of_point. batch_graph: batch
        num_points = x.size(2)
        batch_size = x.size(0)
        assert (x.size(0)==len(batch_graph))
        A_x = torch.zeros(x.size())
        aa = torch.zeros(num_points, 16)

        if x.is_cuda:
            A_x = A_x.cuda()
        for b in range(batch_size):
            bth_graph = batch_graph[b]
            index_b = bth_graph.nonzero()
            x_batch = x[b,:,:]      # channel, num_of_point
            x_batch = x_batch.transpose(0,1)    # num_of_point, channel

            for i in range(num_points):
                idx = index_b[0] == i
                ele = index_b[1][idx]
                rand_idx  = np.random.choice(len(ele),16,replace=False)
                ele = ele[rand_idx]
                aa[i,:] = torch.from_numpy(ele)

            aa = aa.to(torch.int64)
            A_batch = x_batch[aa]   # num_of_point,16,channel
            if x.is_cuda:
                A_batch = A_batch.cuda()
            A_batch = torch.max(A_batch,dim = 1, keepdim=False)[0] # num_of_point,channel
            A_x[b,:,:] = A_batch.transpose(0,1)


            # for i in range(num_points):
            #     i_nb_index = bth_graph[i, :].nonzero()[1]  # the ith point's neighbors' index
            #     A_x[b, :, i] = torch.max(x[b:b+1, :, i_nb_index], dim=2, keepdim=True)[0].view(-1)  # the output size should be 1,channels,1

        A_x = torch.max(A_x, x)     # compare to itself

        return A_x  # batch,channel,num of point

class FoldingNetEnc_with_graph(nn.Module):
    def __init__(self):
        super(FoldingNetEnc_with_graph, self).__init__()
        self.conv1 = torch.nn.Conv1d(12, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 512)

        self.graph_pooling = Graph_Pooling()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)


    def forward(self, x, Cov, batch_graph): # x: batch,3,n; Cov: batch,9,n; batch_graph: batch * scipy.sparse.csr_matrix

        x_cov = torch.cat((x, Cov), 1)   # x_cov: batch,12,n
        x_cov = F.relu(self.bn1(self.conv1(x_cov)))
        x_cov = F.relu(self.bn2(self.conv2(x_cov)))
        x_cov = F.relu(self.bn3(self.conv3(x_cov)))     # x_cov : batch,64,n

        # A_x = torch.zeros(x_cov.size())
        # if x_cov.is_cuda:
        #     A_x = A_x.cuda()
        # for b in range(batch_size):
        #     bth_graph = batch_graph[b]
        #     for i in range(num_points):
        #         i_nb_index = bth_graph[i, :].nonzero()[0]   # the ith point's neighbors' index
        #         A_x[b,:,i] = torch.max(x_cov[b, :, i_nb_index], dim = 2, keepdim=True)[0]  # the output size should be 1,64,1

        A_x = self.graph_pooling(x_cov, batch_graph)   # A_x: batch,64,n
        A_x = F.relu(A_x)
        A_x = F.relu(self.bn4(self.conv4(A_x)))     # batch,128,n
        A_x = F.relu(self.graph_pooling(A_x, batch_graph))   # batch,128,n
        A_x = self.bn5(self.conv5(A_x))    # batch,1024,n
        A_x = torch.max(A_x, dim=2, keepdim=True)[0]     # batch,1024,1
        A_x = A_x.view(-1,1024)     # batch,1024
        A_x = F.relu(self.bn6(self.fc1(A_x)))   # batch,512
        A_x = self.fc2(A_x)     # batch,512

        return A_x


class FoldingNetDecFold1(nn.Module):
    def __init__(self):
        super(FoldingNetDecFold1, self).__init__()
        self.conv1 = nn.Conv1d(514, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)

        self.relu = nn.ReLU()

    def forward(self, x):  # input x = batch,514,45^2
        x = self.relu(self.conv1(x))  # x = batch,512,45^2
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x


class FoldingNetDecFold2(nn.Module):
    def __init__(self):
        super(FoldingNetDecFold2, self).__init__()
        self.conv1 = nn.Conv1d(515, 512, 1)
        self.conv2 = nn.Conv1d(512, 512, 1)
        self.conv3 = nn.Conv1d(512, 3, 1)
        self.relu = nn.ReLU()

    def forward(self, x):  # input x = batch,515,45^2
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


def GridSamplingLayer(batch_size, meshgrid):
    '''
    output Grid points as a NxD matrix
    params = {
    'batch_size': 8
    'meshgrid': [[-0.3,0.3,45],[-0.3,0.3,45]]
    }
    '''

    ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in meshgrid])
    ndim = len(meshgrid)
    grid = np.zeros((np.prod([it[2] for it in meshgrid]), ndim), dtype=np.float32)  # MxD
    for d in range(ndim):
        grid[:, d] = np.reshape(ret[d], -1)
    g = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)

    return g


class FoldingNetDec(nn.Module):
    def __init__(self, w, h):
        self.w = w
        self.h = h
        super(FoldingNetDec, self).__init__()
        self.conv = nn.Conv1d(1024,512,1)
        self.fold1 = FoldingNetDecFold1()
        self.fold2 = FoldingNetDecFold2()

    def forward(self, x):  # input x = batch, 1024

        batch_size = x.size(0)
        x = x.view(-1,1024,1)
        x = self.conv(x) # x=batch, 512,1
        x = x.view(-1,512)
        x = torch.unsqueeze(x, 1)  # x = batch,1,512
        x = x.repeat(1, self.w*self.h, 1)  # x = batch,45^2,512
        code = x
        code = x.transpose(2, 1)  # x = batch,512,45^2

        meshgrid = [[-0.3, 0.3, self.w], [-0.3, 0.3, self.h]]
        grid = GridSamplingLayer(batch_size, meshgrid)  # grid = batch,45^2,2
        grid = torch.from_numpy(grid)

        if x.is_cuda:
            grid = grid.cuda()

        x = torch.cat((x, grid), 2)  # x = batch,45^2,514
        x = x.transpose(2, 1)  # x = batch,514,45^2

        x = self.fold1(x)  # x = batch,3,45^2
        p1 = x  # to observe

        x = torch.cat((code, x), 1)  # x = batch,515,45^2

        x = self.fold2(x)  # x = batch,3,45^2

        #return x, p1
        return x

class Quantization(Function):
    #def __init__(self):
     #   super(Quantization, self).__init__()

    @staticmethod
    def forward(ctx, input):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx,grad_output):
        return grad_output

class Quantization_module(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return Quantization.apply(input)

class FoldingNet(nn.Module):
    def __init__(self):
        super(FoldingNet, self).__init__()
        self.encoder = FoldingNetEnc()
        self.decoder = FoldingNetDec()
        self.quan = Quantization_module()

    def forward(self, x):  # input x = batch,3,number of points
        code, tran = self.encoder(x)  # code = batch,512
        code = self.quan(code)          # quantization

        '''if self.training == 0:      # if now is evaluation, save code
            try:
                os.makedirs('bin')
            except OSError:
                pass
            code_save = code.cpu().detach()
            code_save = code_save.numpy()
            code_save = code_save.astype(int)
            np.savetxt('./bin/test.bin', code_save)
        '''

        x, x_middle = self.decoder(code)  # x = batch,3,45^2

        return x, x_middle,code

class FoldingNet_graph(nn.Module):
    def __init__(self):
        super(FoldingNet_graph, self).__init__()
        self.encoder = FoldingNetEnc_with_graph()
        self.decoder = FoldingNetDec()
        self.quan = Quantization_module()

    def forward(self, x, Cov, batch_graph):
        '''
        x: batch,3,n; Cov: batch,9,n; batch_graph: batch * scipy.sparse.csr_matrix
        '''
        code = self.encoder(x,Cov,batch_graph)
        code = self.quan(code)
        x, x_middle = self.decoder(code)  # x = batch,3,45^2

        return x, x_middle, code





def ChamferDistance(x, y):  # for example, x = batch,2025,3 y = batch,2048,3
    #   compute chamfer distance between tow point clouds x and y

    x_size = x.size()
    y_size = y.size()
    assert (x_size[0] == y_size[0])
    assert (x_size[2] == y_size[2])
    x = torch.unsqueeze(x, 1)  # x = batch,1,2025,3
    y = torch.unsqueeze(y, 2)  # y = batch,2048,1,3

    x = x.repeat(1, y_size[1], 1, 1)  # x = batch,2048,2025,3
    y = y.repeat(1, 1, x_size[1], 1)  # y = batch,2048,2025,3

    x_y = x - y
    x_y = torch.pow(x_y, 2)  # x_y = batch,2048,2025,3
    x_y = torch.sum(x_y, 3, keepdim=True)  # x_y = batch,2048,2025,1
    x_y = torch.squeeze(x_y, 3)  # x_y = batch,2048,2025
    x_y_row, _ = torch.min(x_y, 1, keepdim=True)  # x_y_row = batch,1,2025
    x_y_col, _ = torch.min(x_y, 2, keepdim=True)  # x_y_col = batch,2048,1

    x_y_row = torch.mean(x_y_row, 2, keepdim=True)  # x_y_row = batch,1,1
    x_y_col = torch.mean(x_y_col, 1, keepdim=True)  # batch,1,1
    x_y_row_col = torch.cat((x_y_row, x_y_col), 2)  # batch,1,2
    chamfer_distance, _ = torch.max(x_y_row_col, 2, keepdim=True)  # batch,1,1
    # chamfer_distance = torch.reshape(chamfer_distance,(x_size[0],-1))  #batch,1
    # chamfer_distance = torch.squeeze(chamfer_distance,1)    # batch
    chamfer_distance = torch.mean(chamfer_distance)
    return chamfer_distance


class ChamferLoss(nn.Module):
    # chamfer distance loss
    def __init__(self):
        super(ChamferLoss, self).__init__()

    def forward(self, x, y):
        return ChamferDistance(x, y)





class PointNetDenseCls(nn.Module):
    def __init__(self, k=2):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feat = PointNetfeat(global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans

