#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv, SCBottleneck


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [256, 512, 1024], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #   dehazed 640, 640, 3
        #---------------------------------------------------#
        outputs = []
        # for k, x in enumerate(inputs):
        for k, x in zip(range(len(inputs)-1), inputs):
            x       = self.stems[k](x)
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#

            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs, inputs[3]


class Decoder(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Decoder, self).__init__()
        #Input: 20*20*512
        # Upsampling
        # inputsize:512*20*20, outputsize:256*40*40
        self.conv_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

        # inputsize:256*40*40(concat feat2), outputsize:128*80*80
        self.conv_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )

        # inputsize:128*80*80(concat feat1), outputsize:64*160*160
        self.conv_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        # Output layer
        self.conv_4 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, 7),
            nn.Tanh()
         )


    def forward(self, x, y, z):
        """Standard forward"""
        # Decoding
        # batch size x 256 x 40 x 40
        c1 = self.conv_1(x)

        # batch size x 512 x 40 x 40
        skip1_de = torch.cat((c1, y), 1)

        # batch size x 128 x 80 x 80
        c2 = self.conv_2(skip1_de)

        # batch size x 256 x 80 x 80
        skip2_de = torch.cat((c2, z), 1)

        # batch size x 64 x 160 x 160
        c3 = self.conv_3(skip2_de)

        # upsample: batch size x 64 x 640 x 640
        c4 = self.upsample(c3)

        # batch size x 3 x 640 x 640
        dehaze = self.conv_4(c4)
        return dehaze


class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)

        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        self.bu_conv2       = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

        self.sc1 = SCBottleneck(128, 128)
        self.sc2 = SCBottleneck(128, 128)
        self.sc3 = SCBottleneck(256, 256)
        self.decoder = Decoder(512, 3)

    def forward(self, input):
        if self.training:
            input, clear_x = input.split((4, 4), dim=0)  # split haze and clear images (Batchsize, Batchsize)
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = self.sc1(P4_upsample)
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)
        # P3_out = self.swt3(P4_upsample)

        #-------------------------------------------#
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        P3_downsample   = self.bu_conv2(P3_out) 
        #-------------------------------------------#
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P3_downsample = self.sc2(P3_downsample)
        P3_downsample   = torch.cat([P3_downsample, P4], 1)
        #-------------------------------------------#
        #   40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P4_out          = self.C3_n3(P3_downsample)
        # P4_out = self.swt4(P3_downsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        P4_downsample   = self.bu_conv1(P4_out)
        #-------------------------------------------#
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        #-------------------------------------------#
        P4_downsample = self.sc3(P4_downsample)
        P4_downsample   = torch.cat([P4_downsample, P5], 1)
        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        P5_out          = self.C3_n4(P4_downsample)
        # P5_out = self.swt5(P5_out)
        dehazing = self.decoder(feat3, feat2, feat1)
        return (P3_out, P4_out, P5_out, dehazing)

class YoloBody(nn.Module):
    def __init__(self, num_classes, phi):
        super().__init__()
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        depth, width    = depth_dict[phi], width_dict[phi]
        depthwise       = True if phi == 'nano' else False 

        self.backbone   = YOLOPAFPN(depth, width, depthwise=depthwise)
        self.head       = YOLOXHead(num_classes, width, depthwise=depthwise)

    def forward(self, x):
        fpn_outs    = self.backbone.forward(x)
        outputs     = self.head.forward(fpn_outs)
        return outputs
