# -*- coding:utf-8 -*-

# @file: segmentator_3d_asymm_spconv.py
import copy

import numpy as np
import spconv.pytorch as spconv
import torch
from torch import nn


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()
        self.conv1 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        self.conv1_2 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        self.conv2 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))
        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = conv3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        self.conv1_2 = conv3x1(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))

        shortcut = self.conv1_2(shortcut)
        shortcut = shortcut.replace_feature(self.act1_2(shortcut.features))
        shortcut = shortcut.replace_feature(self.bn0_2(shortcut.features))

        resA = self.conv2(x)
        resA = resA.replace_feature(self.act2(resA.features))
        resA = resA.replace_feature(self.bn1(resA.features))

        resA = self.conv3(resA)
        resA = resA.replace_feature(self.act3(resA.features))
        resA = resA.replace_feature(self.bn2(resA.features))

        resA = resA.replace_feature(resA.features + shortcut.features)

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()
        # self.drop_out = drop_out
        self.trans_dilao = conv3x3(in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        self.conv2 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        self.conv3 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)
        upA = upA.replace_feature(self.trans_act(upA.features))
        upA = upA.replace_feature(self.trans_bn(upA.features))

        ## upsample
        upA = self.up_subm(upA)

        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)
        upE = upE.replace_feature(self.act1(upE.features))
        upE = upE.replace_feature(self.bn1(upE.features))

        upE = self.conv2(upE)
        upE = upE.replace_feature(self.act2(upE.features))
        upE = upE.replace_feature(self.bn2(upE.features))

        upE = self.conv3(upE)
        upE = upE.replace_feature(self.act3(upE.features))
        upE = upE.replace_feature(self.bn3(upE.features))

        return upE


class ReconBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = shortcut.replace_feature(self.bn0(shortcut.features))
        shortcut = shortcut.replace_feature(self.act1(shortcut.features))

        shortcut2 = self.conv1_2(x)
        shortcut2 = shortcut2.replace_feature(self.bn0_2(shortcut2.features))
        shortcut2 = shortcut2.replace_feature(self.act1_2(shortcut2.features))

        shortcut3 = self.conv1_3(x)
        shortcut3 = shortcut3.replace_feature(self.bn0_3(shortcut3.features))
        shortcut3 = shortcut3.replace_feature(self.act1_3(shortcut3.features))
        shortcut = shortcut.replace_feature(shortcut.features + shortcut2.features + shortcut3.features)

        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut

class ReconBlock_dropout(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock_dropout, self).__init__()
        self.conv1 = conv3x1x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)

        self.conv1_2 = conv1x3x1(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)

        self.conv1_3 = conv1x1x3(in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()
        self.dropout3 = nn.Dropout(p=0.5, inplace=False)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut.features = self.bn0(shortcut.features)
        shortcut.features = self.act1(shortcut.features)
        shortcut.features = self.dropout1(shortcut.features)

        shortcut2 = self.conv1_2(x)
        shortcut2.features = self.bn0_2(shortcut2.features)
        shortcut2.features = self.act1_2(shortcut2.features)
        shortcut2.features = self.dropout2(shortcut2.features)

        shortcut3 = self.conv1_3(x)
        shortcut3.features = self.bn0_3(shortcut3.features)
        shortcut3.features = self.act1_3(shortcut3.features)
        shortcut3.features = self.dropout3(shortcut3.features)
        shortcut.features = shortcut.features + shortcut2.features + shortcut3.features

        shortcut.features = shortcut.features * x.features

        return shortcut


class rpl_residual_block(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16,
                 residual_block=None):
        super(rpl_residual_block, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape


        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")
        self.ReconNet_dropout = ReconBlock_dropout(2 * init_size, 2 * init_size, indice_key="recon_dropout")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

    def forward(self, voxel_features, coors, batch_size):
        raise NotImplementedError()
        return


    def forward_no_logits(self, down4c, down4b, down3b, down2b, down1b):

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)
        up0e = self.ReconNet(up1e)
        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        return up0e


class RPL_Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16,
                 residual_block=None):
        super(RPL_Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")
        self.ReconNet_dropout = ReconBlock_dropout(2 * init_size, 2 * init_size, indice_key="recon_dropout")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

        self.residual_block = residual_block


    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)


        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))
        logits = self.logits(up0e)
        y = logits.dense()

        return y

    def forward_no_logits(self, voxel_features, coors, batch_size, PROCESS_BLOCK_3=False, PROCESS_BLOCK_4=False):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        # Residual block
        if self.residual_block is not None:
            residual_up0e = self.residual_block.forward_no_logits(down4c, down4b, down3b, down2b, down1b)
        else:
            residual_up0e = None

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)
        up0e = self.ReconNet(up1e)
        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

        return up0e, residual_up0e




    def forward_dummy(self, voxel_features, coors, batch_size, ood_num):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        index_perm = torch.randperm(voxel_features.shape[0])
        alpha = 1
        beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
        voxel_shuffle = beta * voxel_features + (1 - beta) * voxel_features[index_perm]
        coor_ori = ret.indices.type(torch.LongTensor)
        coor_perm = coor_ori[index_perm]

        up0e_in = self.forward_no_logits(voxel_shuffle, coors, batch_size)
        logits_in = self.logits(up0e_in)
        y_in = logits_in.dense()
        y_out = self.logits2(up0e_in).dense()
        y_out_max, _ = torch.max(y_out, dim = 1, keepdim = True)
        y_ood = torch.cat([y_in, y_out_max], dim = 1)

        up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size)
        logits_in_normal = self.logits(up0e_normal)
        y_in_normal = logits_in_normal.dense()
        y_out_normal = self.logits2(up0e_normal).dense()
        y_normal = torch.cat([y_in_normal, y_out_normal], dim=1)

        y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim = True)
        y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)

        return y_ood, coor_ori, coor_perm, y_normal, y_normal_dummy

    def forward_dummy_2(self, voxel_features, coors, batch_size, ood_num, point_label_tensor):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        coor_ori = coors.type(torch.LongTensor)
        voxel_label_origin = point_label_tensor[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()
        cls_in_points= torch.unique(voxel_label_origin)
        features_cls = torch.zeros(cls_in_points.shape[0]-1,voxel_features.shape[1]).cuda()
        cls_map = torch.zeros(cls_in_points.shape[0]-1)

        cls_tmp = 0
        for cls in cls_in_points:
            if cls:
                features_cls[cls_tmp] = torch.mean(voxel_features[voxel_label_origin==cls], dim=0)
                cls_map[cls_tmp] = cls
                cls_tmp += 1

        index_perm = torch.randperm(cls_map.shape[0])
        alpha = 1
        beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
        features_shuffle = beta * features_cls + (1 - beta) * features_cls[index_perm]

        voxel_shuffle = voxel_features.clone()
        for i in range(cls_map.shape[0]):
            voxel_shuffle[voxel_label_origin==cls_map[i]] = features_shuffle[i]

        up0e_in = self.forward_no_logits(voxel_shuffle, coors, batch_size)
        logits_in = self.logits(up0e_in)
        y_in = logits_in.dense()
        y_out = self.logits2(up0e_in).dense()
        y_out_max, _ = torch.max(y_out, dim = 1, keepdim = True)
        y_ood = torch.cat([y_in, y_out_max], dim = 1)

        up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size)
        logits_in_normal = self.logits(up0e_normal)
        y_in_normal = logits_in_normal.dense()
        y_out_normal = self.logits2(up0e_normal).dense()
        y_normal = torch.cat([y_in_normal, y_out_normal], dim=1)

        y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim = True)
        y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)

        return y_ood, coor_ori, y_normal, y_normal_dummy

    def forward_dummy_3(self, voxel_features, coors, batch_size, ood_num):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        index_perm = torch.randperm(voxel_features.shape[0])
        alpha = 1
        beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
        voxel_shuffle = beta * voxel_features + (1 - beta) * voxel_features[index_perm]
        coor_ori = ret.indices.type(torch.LongTensor)
        coor_perm = coor_ori[index_perm]
        coor_shuffle = beta * coor_ori + (1 - beta) * coor_perm
        coor_shuffle = coor_shuffle.round().type(torch.LongTensor)

        up0e_in = self.forward_no_logits(voxel_shuffle, coor_shuffle, batch_size)
        logits_in = self.logits(up0e_in)
        y_in = logits_in.dense()
        y_out = self.logits2(up0e_in).dense()
        y_out_max, _ = torch.max(y_out, dim = 1, keepdim = True)
        y_ood = torch.cat([y_in, y_out_max], dim = 1)

        up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size)
        logits_in_normal = self.logits(up0e_normal)
        y_in_normal = logits_in_normal.dense()
        y_out_normal = self.logits2(up0e_normal).dense()
        y_normal = torch.cat([y_in_normal, y_out_normal], dim=1)

        y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim = True)
        y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)

        return y_ood, coor_ori, coor_shuffle, y_normal, y_normal_dummy

    def forward_dummy_4(self, voxel_features, coors, batch_size, ood_num):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()

        up0e_in = self.forward_no_logits(voxel_features, coors, batch_size)

        up0e_in_dense = up0e_in.dense().permute(0,2,3,4,1)
        coor_ori = coors.type(torch.LongTensor)
        up0e_in_dense = up0e_in_dense[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()
        alpha = 1
        beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
        index_perm = torch.randperm(up0e_in_dense.shape[0])
        up0e_in_dense_shuffle = beta * up0e_in_dense + (1 - beta) * up0e_in_dense[index_perm]
        coor_perm = coor_ori[index_perm]
        up0e_in_new = spconv.SparseConvTensor(up0e_in_dense_shuffle, coors, self.sparse_shape,
                                      batch_size)

        logits_in = self.logits(up0e_in_new)
        y_in = logits_in.dense()
        y_out = self.logits2(up0e_in_new).dense()
        y_out_max, _ = torch.max(y_out, dim = 1, keepdim = True)
        y_ood = torch.cat([y_in, y_out_max], dim = 1)

        up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size)
        logits_in_normal = self.logits(up0e_normal)
        y_in_normal = logits_in_normal.dense()
        y_out_normal = self.logits2(up0e_normal).dense()
        y_normal = torch.cat([y_in_normal, y_out_normal], dim=1)

        y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim = True)
        y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)

        return y_ood, coor_ori, coor_perm, y_normal, y_normal_dummy

    def forward_dummy_final(self, voxel_features, coors, batch_size, ood_num, PROCESS_BLOCK_3=False, PROCESS_BLOCK_4=False):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        coor_ori = coors.type(torch.LongTensor)

        up0e_normal, residual_up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size, PROCESS_BLOCK_3=PROCESS_BLOCK_3,
                                             PROCESS_BLOCK_4=PROCESS_BLOCK_4)

        # Main model only
        logits_in_normal = self.logits(up0e_normal)
        y_in_normal = logits_in_normal.dense()
        y_out_normal = self.logits2(up0e_normal).dense()
        y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim=True)
        y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)

        # Main model + Residual module
        if residual_up0e_normal is not None:
            combined_up0e_normal = up0e_normal.replace_feature(up0e_normal.features + residual_up0e_normal.features)
            combined_logits_in_normal = self.logits(combined_up0e_normal)
            combined_y_in_normal = combined_logits_in_normal.dense()
            combined_y_out_normal = self.logits2(combined_up0e_normal).dense()
            combined_y_out_normal_2, _ = torch.max(combined_y_out_normal, dim=1, keepdim=True)
            combined_y_normal_dummy = torch.cat([combined_y_in_normal, combined_y_out_normal_2], dim=1)
        else:
            combined_y_normal_dummy = None

        return coor_ori, y_normal_dummy, combined_y_normal_dummy

    def forward_dummy_upper(self, voxel_features, coors, batch_size, ood_num):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        index_perm = torch.randperm(voxel_features.shape[0])
        alpha = 1
        beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
        voxel_shuffle = beta * voxel_features + (1 - beta) * voxel_features[index_perm]
        coor_ori = ret.indices.type(torch.LongTensor)
        coor_perm = coor_ori[index_perm]

        up0e_in = self.forward_no_logits(voxel_shuffle, coors, batch_size)
        logits_in = self.logits(up0e_in)
        y_in = logits_in.dense()
        y_out = self.logits2(up0e_in).dense()
        y_out_max, _ = torch.max(y_out, dim = 1, keepdim = True)
        y_ood = torch.cat([y_in, y_out_max], dim = 1)

        up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size)
        logits_in_normal = self.logits(up0e_normal)
        y_in_normal = logits_in_normal.dense()
        y_out_normal = self.logits2(up0e_normal).dense()
        y_normal = torch.cat([y_in_normal, y_out_normal], dim=1)
        y_out_normal = torch.cat([y_out_normal, y_in_normal[:,5].unsqueeze(1)], dim=1)

        y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim = True)
        y_in_normal[:, 5] = -1e9
        y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)

        return y_ood, coor_ori, coor_perm, y_normal, y_normal_dummy

    def forward_DML(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        coor_ori = coors.type(torch.LongTensor)

        up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size)
        logits_in_normal = self.logits(up0e_normal)
        y_in_normal = logits_in_normal.dense()  # [bs, num_cls, 480, 360, 32]

        features = y_in_normal.permute(0, 2, 3, 4, 1).contiguous()
        features_shape = features.size()
        output = torch.zeros_like(features).cuda()

        centers = torch.zeros(features_shape[-1], features_shape[-1]).cuda()
        magnitude = 1
        for i in range(features_shape[-1]):
            centers[i][i] = magnitude

        for i in range(features_shape[-1]):
            dists = features - centers[i]
            dist2mean = -torch.sum(dists ** 2, -1)
            output[..., i] = dist2mean
        output = output.permute(0, 4, 1, 2, 3)

        return output

    def forward_dropout(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)


        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet_dropout(up1e)

        up0e.features = torch.cat((up0e.features, up1e.features), 1)
        logits = self.logits(up0e)
        y = logits.dense()

        return y

    def forward_dropout_eval(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)


        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        repeat_cnt = 5
        y_final = None
        for i in range(repeat_cnt):
            up0e = self.ReconNet_dropout(up1e)

            up0e.features = torch.cat((up0e.features, up1e.features), 1)
            logits = self.logits(up0e)
            y = logits.dense()
            if y_final is None:
                y_final = y
            else:
                y_final = torch.cat([y_final, y], dim=0)

        return y_final

    def forward_incremental(self, voxel_features, coors, batch_size, incre_cls=None):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        coor_ori = coors.type(torch.LongTensor)

        up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size)
        logits_in_normal = self.logits(up0e_normal)
        y_in = logits_in_normal.dense()
        y_out = self.logits2(up0e_normal).dense()

        if incre_cls == None:
            y_out_incre= y_out[:,0,...].unsqueeze(1)
            y_out_dummy, _ = torch.max(y_out[:,1:,...], dim=1, keepdim=True)
            y_out_dummy = torch.cat([y_in, y_out_dummy], dim=1)
            y_out_dummy = torch.cat([y_out_dummy, y_out_incre], dim=1)
            y_eval = torch.cat([y_in, y_out_incre], dim=1)
        else:
            unknown_clss = [1, 5, 8, 9]
            idx = unknown_clss.index(incre_cls) + 1

            if idx != 1:
                y_out_incre = y_out[:, 0:idx, ...]
            else:
                y_out_incre = y_out[:, 0, ...].unsqueeze(1)
            if idx != 4:
                y_out_dummy, _ = torch.max(y_out[:, idx:, ...], dim=1, keepdim=True)
            else:
                y_out_dummy = y_out[:,4,...].unsqueeze(1)

            y_out_dummy = torch.cat([y_in, y_out_dummy], dim=1)
            y_out_dummy = torch.cat([y_out_dummy, y_out_incre], dim=1)
            y_eval = torch.cat([y_in, y_out_incre], dim=1)

        return coor_ori, y_in, y_out_dummy, y_eval















class Asymm_3d_spconv(nn.Module):
    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):
        super(Asymm_3d_spconv, self).__init__()
        self.nclasses = nclasses
        self.nheight = n_height
        self.strict = False

        sparse_shape = np.array(output_shape)
        # sparse_shape[0] = 11
        print(sparse_shape)
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(num_input_features, init_size, indice_key="pre")
        self.resBlock2 = ResBlock(init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        self.upBlock0 = UpBlock(16 * init_size, 16 * init_size, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(2 * init_size, 2 * init_size, indice_key="recon")
        self.ReconNet_dropout = ReconBlock_dropout(2 * init_size, 2 * init_size, indice_key="recon_dropout")

        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)


        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))
        logits = self.logits(up0e)
        y = logits.dense()

        return y

    def forward_no_logits(self, voxel_features, coors, batch_size, PROCESS_BLOCK_3=False, PROCESS_BLOCK_4=False):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)


        # Perform Singular Value Decomposition on the output of Block 4
        PROCESS_BLOCK_3 = PROCESS_BLOCK_3
        PROCESS_BLOCK_4 = PROCESS_BLOCK_4

        if PROCESS_BLOCK_3 and PROCESS_BLOCK_4:

            # Remove from the outputs of Block 3
            feat1 = copy.deepcopy(down3b.features.permute(1, 0))
            u, s, v = torch.linalg.svd(feat1, full_matrices=False)
            u = u.unsqueeze(0)
            s = s.unsqueeze(0)
            v = v.unsqueeze(0)
            feat1 = feat1 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
            feat1 = feat1.squeeze(0)
            feat1 = feat1.permute(1, 0)

            down3b_early_removal = copy.deepcopy(down3b).replace_feature(feat1)
            down3c_early_removal = self.resBlock4.pool(down3b_early_removal)
            down4c_early_removal, down4b_early_removal = self.resBlock5(down3c_early_removal)


            up4e_early_removal = self.upBlock0(down4c_early_removal, down4b_early_removal)
            up3e_early_removal = self.upBlock1(up4e_early_removal, down3b_early_removal)
            up2e_early_removal = self.upBlock2(up3e_early_removal, down2b)
            up1e_early_removal = self.upBlock3(up2e_early_removal, down1b)
            up0e_early_removal = self.ReconNet(up1e_early_removal)
            up0e_early_removal = up0e_early_removal.replace_feature(torch.cat((up0e_early_removal.features, up1e_early_removal.features), 1))

            ############################################################
            # Remove from the outputs of Block 4

            down4c, down4b = self.resBlock5(down3c)
            feat2 = copy.deepcopy(down4b.features.permute(1, 0))
            u, s, v = torch.linalg.svd(feat2, full_matrices=False)
            u = u.unsqueeze(0)
            s = s.unsqueeze(0)
            v = v.unsqueeze(0)
            feat2 = feat2 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
            feat2 = feat2.squeeze(0)
            feat2 = feat2.permute(1, 0)

            down4b_late_removal = copy.deepcopy(down4b).replace_feature(feat2)
            down4c_late_removal = self.resBlock5.pool(down4b_late_removal)

            up4e_late_removal = self.upBlock0(down4c_late_removal, down4b_late_removal)
            up3e_late_removal = self.upBlock1(up4e_late_removal, down3b)
            up2e_late_removal = self.upBlock2(up3e_late_removal, down2b)
            up1e_late_removal = self.upBlock3(up2e_late_removal, down1b)
            up0e_late_removal = self.ReconNet(up1e_late_removal)
            up0e_late_removal = up0e_late_removal.replace_feature(torch.cat((up0e_late_removal.features, up1e_late_removal.features), 1))

            return [up0e_early_removal, up0e_late_removal]

        elif PROCESS_BLOCK_3:

            raise NotImplementedError('Not implemented yet.')

            feat1 = copy.deepcopy(down3c.features.permute(1, 0))
            u, s, v = torch.linalg.svd(feat1, full_matrices=False)
            u = u.unsqueeze(0)
            s = s.unsqueeze(0)
            v = v.unsqueeze(0)
            feat1 = feat1 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
            feat1 = feat1.squeeze(0)
            feat1 = feat1.permute(1, 0)

            processed_down3c = copy.deepcopy(down3c).replace_feature(feat1)

            up4e = self.upBlock0(processed_down3c, down3b)
            up3e = self.upBlock1(up4e, down3b)
            up2e = self.upBlock2(up3e, down2b)
            up1e = self.upBlock3(up2e, down1b)
            up0e = self.ReconNet(up1e)
            up0e_from_block_3 = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

            return up0e_from_block_3

        elif PROCESS_BLOCK_4:

            raise NotImplementedError('Not implemented yet.')

            down4c, down4b = self.resBlock5(down3c)
            feat2 = copy.deepcopy(down4c.features.permute(1, 0))
            u, s, v = torch.linalg.svd(feat2, full_matrices=False)
            u = u.unsqueeze(0)
            s = s.unsqueeze(0)
            v = v.unsqueeze(0)
            feat2 = feat2 - s[:, 0:1].unsqueeze(2) * u[:, :, 0:1].bmm(v[:, 0:1, :])
            feat2 = feat2.squeeze(0)
            feat2 = feat2.permute(1, 0)

            processed_down4c = copy.deepcopy(down4c).replace_feature(feat2)

            up4e = self.upBlock0(processed_down4c, down4b)
            up3e = self.upBlock1(up4e, down3b)
            up2e = self.upBlock2(up3e, down2b)
            up1e = self.upBlock3(up2e, down1b)
            up0e = self.ReconNet(up1e)
            up0e_from_block_4 = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

            return up0e_from_block_4

        else:#* here ... 
            down4c, down4b = self.resBlock5(down3c)

            up4e = self.upBlock0(down4c, down4b)
            up3e = self.upBlock1(up4e, down3b)
            up2e = self.upBlock2(up3e, down2b)
            up1e = self.upBlock3(up2e, down1b)
            up0e = self.ReconNet(up1e)
            up0e = up0e.replace_feature(torch.cat((up0e.features, up1e.features), 1))

            return up0e

    def forward_dummy(self, voxel_features, coors, batch_size, ood_num):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        index_perm = torch.randperm(voxel_features.shape[0])
        alpha = 1
        beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
        voxel_shuffle = beta * voxel_features + (1 - beta) * voxel_features[index_perm]
        coor_ori = ret.indices.type(torch.LongTensor)
        coor_perm = coor_ori[index_perm]

        up0e_in = self.forward_no_logits(voxel_shuffle, coors, batch_size)
        logits_in = self.logits(up0e_in)
        y_in = logits_in.dense()
        y_out = self.logits2(up0e_in).dense()
        y_out_max, _ = torch.max(y_out, dim = 1, keepdim = True)
        y_ood = torch.cat([y_in, y_out_max], dim = 1)

        up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size)
        logits_in_normal = self.logits(up0e_normal)
        y_in_normal = logits_in_normal.dense()
        y_out_normal = self.logits2(up0e_normal).dense()
        y_normal = torch.cat([y_in_normal, y_out_normal], dim=1)

        y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim = True)
        y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)

        return y_ood, coor_ori, coor_perm, y_normal, y_normal_dummy

    def forward_dummy_2(self, voxel_features, coors, batch_size, ood_num, point_label_tensor):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        coor_ori = coors.type(torch.LongTensor)
        voxel_label_origin = point_label_tensor[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()
        cls_in_points= torch.unique(voxel_label_origin)
        features_cls = torch.zeros(cls_in_points.shape[0]-1,voxel_features.shape[1]).cuda()
        cls_map = torch.zeros(cls_in_points.shape[0]-1)

        cls_tmp = 0
        for cls in cls_in_points:
            if cls:
                features_cls[cls_tmp] = torch.mean(voxel_features[voxel_label_origin==cls], dim=0)
                cls_map[cls_tmp] = cls
                cls_tmp += 1

        index_perm = torch.randperm(cls_map.shape[0])
        alpha = 1
        beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
        features_shuffle = beta * features_cls + (1 - beta) * features_cls[index_perm]

        voxel_shuffle = voxel_features.clone()
        for i in range(cls_map.shape[0]):
            voxel_shuffle[voxel_label_origin==cls_map[i]] = features_shuffle[i]

        up0e_in = self.forward_no_logits(voxel_shuffle, coors, batch_size)
        logits_in = self.logits(up0e_in)
        y_in = logits_in.dense()
        y_out = self.logits2(up0e_in).dense()
        y_out_max, _ = torch.max(y_out, dim = 1, keepdim = True)
        y_ood = torch.cat([y_in, y_out_max], dim = 1)

        up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size)
        logits_in_normal = self.logits(up0e_normal)
        y_in_normal = logits_in_normal.dense()
        y_out_normal = self.logits2(up0e_normal).dense()
        y_normal = torch.cat([y_in_normal, y_out_normal], dim=1)

        y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim = True)
        y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)

        return y_ood, coor_ori, y_normal, y_normal_dummy

    def forward_dummy_3(self, voxel_features, coors, batch_size, ood_num):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        index_perm = torch.randperm(voxel_features.shape[0])
        alpha = 1
        beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
        voxel_shuffle = beta * voxel_features + (1 - beta) * voxel_features[index_perm]
        coor_ori = ret.indices.type(torch.LongTensor)
        coor_perm = coor_ori[index_perm]
        coor_shuffle = beta * coor_ori + (1 - beta) * coor_perm
        coor_shuffle = coor_shuffle.round().type(torch.LongTensor)

        up0e_in = self.forward_no_logits(voxel_shuffle, coor_shuffle, batch_size)
        logits_in = self.logits(up0e_in)
        y_in = logits_in.dense()
        y_out = self.logits2(up0e_in).dense()
        y_out_max, _ = torch.max(y_out, dim = 1, keepdim = True)
        y_ood = torch.cat([y_in, y_out_max], dim = 1)

        up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size)
        logits_in_normal = self.logits(up0e_normal)
        y_in_normal = logits_in_normal.dense()
        y_out_normal = self.logits2(up0e_normal).dense()
        y_normal = torch.cat([y_in_normal, y_out_normal], dim=1)

        y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim = True)
        y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)

        return y_ood, coor_ori, coor_shuffle, y_normal, y_normal_dummy

    def forward_dummy_4(self, voxel_features, coors, batch_size, ood_num):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()

        up0e_in = self.forward_no_logits(voxel_features, coors, batch_size)

        up0e_in_dense = up0e_in.dense().permute(0,2,3,4,1)
        coor_ori = coors.type(torch.LongTensor)
        up0e_in_dense = up0e_in_dense[coor_ori.permute(1, 0).chunk(chunks=4, dim=0)].squeeze()
        alpha = 1
        beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
        index_perm = torch.randperm(up0e_in_dense.shape[0])
        up0e_in_dense_shuffle = beta * up0e_in_dense + (1 - beta) * up0e_in_dense[index_perm]
        coor_perm = coor_ori[index_perm]
        up0e_in_new = spconv.SparseConvTensor(up0e_in_dense_shuffle, coors, self.sparse_shape,
                                      batch_size)

        logits_in = self.logits(up0e_in_new)
        y_in = logits_in.dense()
        y_out = self.logits2(up0e_in_new).dense()
        y_out_max, _ = torch.max(y_out, dim = 1, keepdim = True)
        y_ood = torch.cat([y_in, y_out_max], dim = 1)

        up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size)
        logits_in_normal = self.logits(up0e_normal)
        y_in_normal = logits_in_normal.dense()
        y_out_normal = self.logits2(up0e_normal).dense()
        y_normal = torch.cat([y_in_normal, y_out_normal], dim=1)

        y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim = True)
        y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)

        return y_ood, coor_ori, coor_perm, y_normal, y_normal_dummy

    def forward_dummy_final(self, voxel_features, coors, batch_size, ood_num, PROCESS_BLOCK_3=False, PROCESS_BLOCK_4=False):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        coor_ori = coors.type(torch.LongTensor)

        PROCESS_BLOCK_3 = PROCESS_BLOCK_3
        PROCESS_BLOCK_4 = PROCESS_BLOCK_4

        if PROCESS_BLOCK_3 and PROCESS_BLOCK_4:
            res = self.forward_no_logits(voxel_features, coors, batch_size, PROCESS_BLOCK_3=PROCESS_BLOCK_3,
                                                 PROCESS_BLOCK_4=PROCESS_BLOCK_4)
            up0e_normal_early_removal, up0e_normal_late_removal = res[0], res[1]

            logits_in_normal_early_removal = self.logits(up0e_normal_early_removal)
            logits_in_normal_late_removal = self.logits(up0e_normal_late_removal)

            # Take average of the two sets of logits
            y_in_normal_early_removal = logits_in_normal_early_removal.dense()
            y_out_normal_early_removal = self.logits2(up0e_normal_early_removal).dense()
            y_out_normal_2_early_removal, _ = torch.max(y_out_normal_early_removal, dim=1, keepdim=True)
            y_normal_dummy_early_removal = torch.cat([y_in_normal_early_removal, y_out_normal_2_early_removal], dim=1)

            y_in_normal_late_removal = logits_in_normal_late_removal.dense()
            y_out_normal_late_removal = self.logits2(up0e_normal_late_removal).dense()
            y_out_normal_2_late_removal, _ = torch.max(y_out_normal_late_removal, dim=1, keepdim=True)
            y_normal_dummy_late_removal = torch.cat([y_in_normal_late_removal, y_out_normal_2_late_removal], dim=1)

            y_normal_dummy = (y_normal_dummy_early_removal + y_normal_dummy_late_removal) / 2

            # TODO
            # temperature = 1
            # conf = temperature * torch.logsumexp(y_normal_dummy / temperature, dim=1, keepdim=True).reshape((-1,1))


        else:
            up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size, PROCESS_BLOCK_3=PROCESS_BLOCK_3,
                                                 PROCESS_BLOCK_4=PROCESS_BLOCK_4)
            logits_in_normal = self.logits(up0e_normal)

            y_in_normal = logits_in_normal.dense()
            y_out_normal = self.logits2(up0e_normal).dense()
            """
                3 dummy classifier but only keep the the maxmum logit, namely, only keep one output as novel  class prediction logit 
            """
            y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim=True)#*
            y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)

        # y_in_normal = logits_in_normal.dense()
        # y_out_normal = self.logits2(up0e_normal).dense()
        #
        # y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim = True)
        # y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)

        return coor_ori, y_normal_dummy

    def forward_dummy_upper(self, voxel_features, coors, batch_size, ood_num):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        index_perm = torch.randperm(voxel_features.shape[0])
        alpha = 1
        beta = torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
        voxel_shuffle = beta * voxel_features + (1 - beta) * voxel_features[index_perm]
        coor_ori = ret.indices.type(torch.LongTensor)
        coor_perm = coor_ori[index_perm]

        up0e_in = self.forward_no_logits(voxel_shuffle, coors, batch_size)
        logits_in = self.logits(up0e_in)
        y_in = logits_in.dense()
        y_out = self.logits2(up0e_in).dense()
        y_out_max, _ = torch.max(y_out, dim = 1, keepdim = True)
        y_ood = torch.cat([y_in, y_out_max], dim = 1)

        up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size)
        logits_in_normal = self.logits(up0e_normal)
        y_in_normal = logits_in_normal.dense()
        y_out_normal = self.logits2(up0e_normal).dense()
        y_normal = torch.cat([y_in_normal, y_out_normal], dim=1)
        y_out_normal = torch.cat([y_out_normal, y_in_normal[:,5].unsqueeze(1)], dim=1)

        y_out_normal_2, _ = torch.max(y_out_normal, dim=1, keepdim = True)
        y_in_normal[:, 5] = -1e9
        y_normal_dummy = torch.cat([y_in_normal, y_out_normal_2], dim=1)

        return y_ood, coor_ori, coor_perm, y_normal, y_normal_dummy

    def forward_DML(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        coor_ori = coors.type(torch.LongTensor)

        up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size)
        logits_in_normal = self.logits(up0e_normal)
        y_in_normal = logits_in_normal.dense()  # [bs, num_cls, 480, 360, 32]

        features = y_in_normal.permute(0, 2, 3, 4, 1).contiguous()
        features_shape = features.size()
        output = torch.zeros_like(features).cuda()

        centers = torch.zeros(features_shape[-1], features_shape[-1]).cuda()
        magnitude = 1
        for i in range(features_shape[-1]):
            centers[i][i] = magnitude

        for i in range(features_shape[-1]):
            dists = features - centers[i]
            dist2mean = -torch.sum(dists ** 2, -1)
            output[..., i] = dist2mean
        output = output.permute(0, 4, 1, 2, 3)

        return output

    def forward_dropout(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)


        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet_dropout(up1e)

        up0e.features = torch.cat((up0e.features, up1e.features), 1)
        logits = self.logits(up0e)
        y = logits.dense()

        return y

    def forward_dropout_eval(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)


        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        repeat_cnt = 5
        y_final = None
        for i in range(repeat_cnt):
            up0e = self.ReconNet_dropout(up1e)

            up0e.features = torch.cat((up0e.features, up1e.features), 1)
            logits = self.logits(up0e)
            y = logits.dense()
            if y_final is None:
                y_final = y
            else:
                y_final = torch.cat([y_final, y], dim=0)

        return y_final

    def forward_incremental(self, voxel_features, coors, batch_size, incre_cls=None):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        coor_ori = coors.type(torch.LongTensor)

        up0e_normal = self.forward_no_logits(voxel_features, coors, batch_size)
        logits_in_normal = self.logits(up0e_normal)
        y_in = logits_in_normal.dense()
        y_out = self.logits2(up0e_normal).dense()

        if incre_cls == None:
            y_out_incre= y_out[:,0,...].unsqueeze(1)
            y_out_dummy, _ = torch.max(y_out[:,1:,...], dim=1, keepdim=True)
            y_out_dummy = torch.cat([y_in, y_out_dummy], dim=1)
            y_out_dummy = torch.cat([y_out_dummy, y_out_incre], dim=1)
            y_eval = torch.cat([y_in, y_out_incre], dim=1)
        else:
            unknown_clss = [1, 5, 8, 9]
            idx = unknown_clss.index(incre_cls) + 1

            if idx != 1:
                y_out_incre = y_out[:, 0:idx, ...]
            else:
                y_out_incre = y_out[:, 0, ...].unsqueeze(1)
            if idx != 4:
                y_out_dummy, _ = torch.max(y_out[:, idx:, ...], dim=1, keepdim=True)
            else:
                y_out_dummy = y_out[:,4,...].unsqueeze(1)

            y_out_dummy = torch.cat([y_in, y_out_dummy], dim=1)
            y_out_dummy = torch.cat([y_out_dummy, y_out_incre], dim=1)
            y_eval = torch.cat([y_in, y_out_incre], dim=1)

        return coor_ori, y_in, y_out_dummy, y_eval