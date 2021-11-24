import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import lib.backbone.s3d as s3d
from lib.backbone.select_backbone import select_resnet


class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x


def get_debug_hook_grad(name):
    def debug_hook_grad(grad):
        print("Debug hook grad {}\n"
              "Has NaN: {}\n"
              "Has inf: {}\n"
              "Has zero: {}\n"
              "Min: {}\n"
              "Max: {}\n".format(
            name,
            torch.any(torch.isnan(grad)),
            torch.any(torch.isinf(grad)),
            torch.any(grad == 0.0),
            torch.min(grad),
            torch.max(grad)
            ))

        return grad

    return debug_hook_grad


class SkeleMotionBackbone(nn.Module):
    def __init__(self, final_width, seq_len=32, debug=False):
        super(SkeleMotionBackbone, self).__init__()

        linear_hidden_map = {30: 1536, 32: 2048}
        self.linear_hidden_width = linear_hidden_map[seq_len]
        self.final_width = final_width

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=1)
        # nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.maxpool2 = nn.MaxPool2d(3, stride=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 5))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        # nn.BatchNorm2d(32),
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        # nn.BatchNorm2d(64),
        self.relu4 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(in_features=self.linear_hidden_width, out_features=self.final_width)
        self.relu5 = nn.ReLU()

        self.linear2 = nn.Linear(in_features=self.final_width, out_features=self.final_width)

        if debug:
            self.linear1.weight.register_hook(get_debug_hook_grad("Linear 1"))
            self.linear2.weight.register_hook(get_debug_hook_grad("Linear 2"))

    def forward(self, skele_motion_data):
        fe = self.conv1(skele_motion_data)
        fe = self.relu1(fe)

        fe = self.conv2(fe)
        fe = self.maxpool2(fe)
        fe = self.relu2(fe)

        fe = self.conv3(fe)
        fe = self.maxpool3(fe)
        fe = self.relu3(fe)

        fe = self.conv4(fe)
        fe = self.maxpool4(fe)
        fe = self.relu4(fe)

        fe = self.flatten(fe)

        fe = self.linear1(fe)
        fe = self.relu5(fe)

        fe = self.linear2(fe)

        return fe


class SkeleContrastS3D(nn.Module):
    '''Module which performs contrastive learning by matching extracted feature
        vectors from the skele-motion skeleton representation and features extracted from RGB videos.'''

    def __init__(self,
                 vid_backbone='s3d',
                 sk_backbone="sk-motion-7",
                 representation_size=512,
                 hidden_width=512,
                 swav_prototype_count=0,
                 seq_len=32,
                 debug=False,
                 random_seed=42):
        super(SkeleContrastS3D, self).__init__()

        torch.cuda.manual_seed(random_seed)

        print("============Model================")
        print('Using SkeleContrastS3D model.')

        self.vid_backbone_name = vid_backbone
        self.sk_backbone_name = sk_backbone

        self.representation_size = representation_size
        self.hidden_width = hidden_width

        self.swav_prototype_count = swav_prototype_count

        self.debug = debug

        if "s3d" in vid_backbone:
            if vid_backbone == "s3d":
                print('The video backbone is the S3D network.')
                self.vid_backbone = s3d.S3D(self.hidden_width)
            else:
                raise ValueError

        # The first linear layer is part of the S3D architecture
        self.vid_fc2 = nn.Sequential(
            nn.BatchNorm1d(self.hidden_width),
            nn.ReLU(),
            nn.Linear(self.hidden_width, self.hidden_width),
            )

        self.vid_fc_rep = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_width, self.representation_size),
            )

        if "sk-motion" in sk_backbone:
            if "sk-motion-7" == self.sk_backbone_name:
                print('The skeleton backbone is the SkeleMotion-7 network.')
                self.sk_backbone = SkeleMotionBackbone(self.hidden_width, seq_len=seq_len)

        self.sk_fc_rep = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden_width, self.representation_size),
            )

        if self.swav_prototype_count is not None and self.swav_prototype_count > 0:
            print(
                f"Using {self.swav_prototype_count} SWAV prototypes.")
            self.prototypes = torch.nn.Parameter(
                torch.randn(self.swav_prototype_count, self.representation_size, requires_grad=True))

        _initialize_weights(self.vid_fc2)
        _initialize_weights(self.vid_fc_rep)

        _initialize_weights(self.sk_backbone)
        _initialize_weights(self.sk_fc_rep)

        if self.swav_prototype_count is not None and self.swav_prototype_count > 0:
            nn.init.orthogonal_(self.prototypes, 1)

        if hasattr(self, "vid_backbone"):
            print(f"The video backbone has "
                  f"{sum(p.numel() for p in self.vid_backbone.parameters() if p.requires_grad)} trainable parameters.")

        if hasattr(self, "sk_backbone"):
            print(f"The skele motion backbone has "
                  f"{sum(p.numel() for p in self.sk_backbone.parameters() if p.requires_grad)} trainable parameters.")

        print(f"This model has {sum(p.numel() for p in self.parameters() if p.requires_grad)} trainable parameters.")
        print("=================================")

    def _forward_sk(self, block_sk):
        (Ba, C, T, J) = block_sk.shape

        features = self.sk_backbone(block_sk)
        features = self.sk_fc_rep(features)

        is_zero = features == 0.0

        zero_row = is_zero.all(dim=1)

        # TODO: this is a dirty hack, weights are nan if a row is scored which is completely zero, find better solution.
        features[zero_row] = 0.00000001  # This prevents the norm of the 0 vector to be nan.
        return features

    def _forward_rgb(self, block_rgb):
        # block: [B, C, SL, W, H] Batch, Num Seq, Channels, Seq Len, Width Height
        ### extract feature ###
        # (B, C, SL, H, W) = block_rgb.shape

        # For the backbone, first dimension is the  batch size -> Blocks are calculated separately.
        feature = self.vid_backbone(block_rgb)  # (B, hidden_width)
        del block_rgb

        feature = self.vid_fc2(feature)
        feature = self.vid_fc_rep(feature)

        return feature

    def forward(self, block_rgbs, block_sk, body_counts=None):
        # block_rgb: (B, V, C, SL, W, H) Batch, View Count, Channels, Seq Len, Height, Width
        # block_sk: (Ba, Bo, C, T, J) Batch, Bodies, Channels, Timestep, Joint

        pred_rgbs, pred_sk, pred_rgb_projs, pred_sk_projs = None, None, None, None

        view_count = None

        if type(block_rgbs) is list:
            view_count = len(block_rgbs)
            pred_rgbs_vs = []

            for block_rgb_view in block_rgbs:
                # bv_shape = block_rgb_view.shape

                bs = len(block_rgb_view)

                pred_rgbs_vs.append(self._forward_rgb(block_rgb_view))

            pred_rgbs = torch.cat(pred_rgbs_vs, dim=0)
        else:
            bs = len(block_rgbs)
            pred_rgbs = self._forward_rgb(block_rgbs)

        pred_rgbs = pred_rgbs.contiguous()
        pred_rgbs = torch.nn.functional.normalize(pred_rgbs)

        if self.swav_prototype_count is not None and self.swav_prototype_count > 0:
            pred_rgb_projs = F.linear(pred_rgbs, self.prototypes, bias=None)
            #  pred_rgb_projs = self.prototypes(pred_rgbs)

        if view_count is not None:  # Multiple views
            pred_rgbs = pred_rgbs.view(bs, view_count, -1)

            if pred_rgb_projs is not None:
                pred_rgb_projs = pred_rgb_projs.view(bs, view_count, -1)

        if block_sk is not None:
            sk_shape = block_sk.shape

            bs = len(block_sk)
            mbc = 1

            if len(sk_shape) > 4:  # Multiple bodies per batch
                bs, mbc, c, t, j = sk_shape
                block_sk = block_sk.view(bs * mbc, c, t, j)

                # We only forward existing bodies, since it would alterate batch normalization otherwise.
                existing_body_selector = [True if i % mbc < body_counts[i // mbc] else False for i in range(bs * mbc)]
                block_sk = block_sk[existing_body_selector]

            pred_sk = self._forward_sk(block_sk)
            pred_sk = pred_sk.contiguous()
            pred_sk = torch.nn.functional.normalize(pred_sk)

            if self.swav_prototype_count is not None and self.swav_prototype_count > 0:
                # pred_sk_projs = self.prototypes(pred_sk)
                pred_sk_projs = F.linear(pred_sk, self.prototypes, bias=None)

            if len(sk_shape) > 4:  # Multiple views
                pred_sk_n = torch.zeros(bs, mbc, pred_sk.shape[-1], device=pred_sk.device)

                bod_idx = 0
                for i, b_c in enumerate(body_counts):
                    pred_sk_n[i, 0: b_c] = pred_sk[bod_idx: bod_idx + b_c]
                    bod_idx += b_c

                pred_sk = pred_sk_n

                if pred_sk_projs is not None:
                    pred_sk_projs_n = torch.zeros(bs, mbc, pred_sk_projs.shape[-1], device=pred_sk_projs.device)

                    bod_idx = 0
                    for i, b_c in enumerate(body_counts):
                        pred_sk_projs_n[i, 0: b_c] = pred_sk_projs[bod_idx: bod_idx + b_c]
                        bod_idx += b_c

                    pred_sk_projs = pred_sk_projs_n

        if self.swav_prototype_count is not None and self.swav_prototype_count > 0:
            return pred_rgbs, pred_sk, pred_rgb_projs, pred_sk_projs
        else:
            return pred_rgbs, pred_sk



def _initialize_weights(module):
    for name, param in module.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name and len(param.shape) > 1:
            nn.init.orthogonal_(param, 1)
