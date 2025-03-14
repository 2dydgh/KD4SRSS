# model.py

import time
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import sys
import numpy as np
import copy

from PIL import Image

from torchinfo import summary
from thop import profile as profile_thop    
from DLCs.FLOPs import profile
from utils.calc_margin import calculate_margin



# Conv 2D with stride=1 & dilation=1
def Conv_(in_c, out_c, k_size=3, groups=1, bias=False, padding_mode='replicate'):
    p_size = int((k_size - 1)//2) # padding size

    if k_size-1 != 2*p_size:
        print("Kernel size should be odd value. Currnet k_size:", k_size)
        sys.exit(9)

    return nn.Conv2d(in_channels=in_c, out_channels=out_c
                    ,kernel_size=k_size, stride=1
                    ,padding=p_size, dilation=1
                    ,groups=groups
                    ,bias=bias, padding_mode=padding_mode
                    )

# Conv 2D with kernel=3 & stride=1
def Conv_3x3(in_c, out_c, d_size=1, groups=1, bias=False, padding_mode='replicate'):
    p_size = int(d_size) # set padding size with dilation value

    return nn.Conv2d(in_channels=in_c, out_channels=out_c
                    ,kernel_size=3, stride=1
                    ,padding=p_size, dilation=d_size
                    ,groups=groups
                    ,bias=bias, padding_mode=padding_mode
                    )



# single-channel aware attention (SCAA) module
class SCAA(nn.Module):
    def __init__(self, in_c, k_size):
        super(SCAA, self).__init__()
        print("SCAA with", k_size, "x", k_size, "conv")
        self.layer_gconv_sig = nn.Sequential(Conv_(in_c, in_c, k_size=k_size, groups=in_c)
                                            ,nn.Sigmoid()
                                            )

    def forward(self, in_x):
        return in_x * self.layer_gconv_sig(in_x)


class fixed_kernel(nn.Module):
    def __init__(self, k_type=None, in_c=3, out_c=3, device=None):
        super(fixed_kernel, self).__init__()
        self.in_c = in_c
        self.out_c = out_c

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if k_type is None:
            k_type = "laplacian"

        if k_type == "laplacian":
            print("fixed_kernel is laplacian")
            k_weight = torch.tensor(np.array([[1, 1, 1]
                                             ,[1,-8, 1]
                                             ,[1, 1, 1]
                                             ])).float().to(device)

        elif k_type == "laplacian_2":
            print("fixed_kernel is laplacian_2")
            k_weight = torch.tensor(np.array([[0, 1, 0]
                                             ,[1,-4, 1]
                                             ,[0, 1, 0]
                                             ])).float().to(device)

        elif k_type == "laplacian_3":
            print("fixed_kernel is laplacian_3")
            k_weight = torch.tensor(np.array([[-1,-1,-1]
                                             ,[-1, 8,-1]
                                             ,[-1,-1,-1]
                                             ])).float().to(device)



        k_weight = k_weight.unsqueeze(dim=0)
        _k_weight = k_weight.clone().detach()


        k_weight = k_weight.unsqueeze(dim=0)
        _k_weight = k_weight.clone().detach()
        while(True):
            _C, _, _, _ = k_weight.shape
            if _C >= out_c:
                break
            else:
                k_weight = torch.cat((k_weight, _k_weight), dim=0)

        self.register_buffer('weight', k_weight)

    def forward(self, in_x):
        return F.conv2d(in_x, self.weight
                       ,bias=None, stride=1, padding="same", dilation=1, groups=self.in_c
                       )

# spatial boundary-aware attention (SBAA) module
class SBAA(nn.Module):
    def __init__(self, in_c):
        super(SBAA, self).__init__()

        self.fixed_conv = fixed_kernel(k_type="laplacian_3", in_c=in_c, out_c=in_c)

        self.train_conv = Conv_(in_c=in_c, out_c=in_c, k_size=3, groups=in_c)

        self.layer_sig  = nn.Sigmoid()


    def forward(self, in_x):

        feat_sig = self.layer_sig(self.fixed_conv(in_x) + self.train_conv(in_x))

        return in_x * feat_sig


def stack_block(block, count, recursive=True):
    stacks = torch.nn.Sequential()

    if recursive:
        for i_count in range(count):
            stacks.add_module(str(i_count), block)
    else:
        for i_count in range(count):
            stacks.add_module(str(i_count), copy.deepcopy(block))

    return stacks

# It's a DUMMY
class DummyModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DummyModule, self).__init__()
        # print("It's a DUMMY")
        self.dummy = True

    def forward(self, in_x):
        # bypass
        return in_x

# spatial boundary aware (SBA) block
class SBA_block(nn.Module):
    def __init__(self, in_c, mid_c, out_c, use_m1=True, use_m2=True):
        super(SBA_block, self).__init__()
        # m1 = SBAA
        # m2 = SCAA
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if use_m1:
            print("m1 = SBAA = True")
            self.layer_1_1 = nn.Sequential(Conv_3x3(in_c//2, mid_c, d_size=1, groups=1)
                                           , SBAA(mid_c)
                                           , Conv_3x3(mid_c, in_c//2, d_size=1, groups=1)
                                           )
        else:
            print("m1 = SCAA =  False")
            self.layer_1_1 = nn.Sequential(Conv_3x3(in_c//2, mid_c, d_size=1, groups=1)
                                          ,DummyModule(mid_c)
                                          ,Conv_3x3(mid_c, in_c//2, d_size=1, groups=1)
                                          )

        self.layer_1_2 = nn.Sequential(Conv_3x3(in_c - in_c//2, mid_c, d_size=1, groups=1)
                                      ,self.act
                                      ,Conv_3x3(mid_c, in_c - in_c//2, d_size=1, groups=1)
                                      )

        if use_m2:
            print("m2 = SCAA = True")
            self.layer_1_3 = SCAA(in_c, 1)
        else:
            print("m2 = SCAA = False")
            self.layer_1_3 = DummyModule(in_c, 1)


    def forward(self, in_x):
        _, _C, _, _ = in_x.shape

        f_1_1 = self.layer_1_1(in_x[:, :_C//2,:,:])
        f_1_2 = self.layer_1_2(in_x[:, _C//2:,:,:])
        f_1_3 = self.layer_1_3(in_x)

        return torch.cat((f_1_1,f_1_2),dim=1) + f_1_3



class model_I2F(nn.Module):
    def __init__(self, in_c, mid_c, basic_blocks, use_m1=True, use_m2=True):
        super(model_I2F, self).__init__()

        self.layer_init = nn.Sequential(Conv_(in_c=in_c, out_c=mid_c, k_size=3, groups=1)
                                       ,
                                       )


        _str = "model with " + str(basic_blocks//3) + " x 3 basic_blocks (total " + str((basic_blocks//3)*3) + ")"
        warnings.warn(_str)
        if use_m1:
            _str = "(m1) SBAA = True "
        else:
            _str = "(m1) SBAA = False"

        if use_m2:
            _str += " / (m2) SCAA = True "
        else:
            _str += " / (m2) SCAA = False"
        warnings.warn(_str)

        self.layer_mid_1 = stack_block(SBA_block(mid_c, mid_c // 2, mid_c, use_m1=use_m1, use_m2=use_m2), count=basic_blocks // 3, recursive=False)
        self.layer_mid_2 = stack_block(SBA_block(mid_c, mid_c // 2, mid_c, use_m1=use_m1, use_m2=use_m2), count=basic_blocks // 3, recursive=False)
        self.layer_mid_3 = stack_block(SBA_block(mid_c, mid_c // 2, mid_c, use_m1=use_m1, use_m2=use_m2), count=basic_blocks // 3, recursive=False)


    def forward(self, in_x):

        f_init_1 = self.layer_init(in_x)

        f_mid_1 = self.layer_mid_1(f_init_1)
        f_mid_2 = self.layer_mid_2(f_mid_1)
        f_mid_3 = self.layer_mid_3(f_mid_2)

        return f_mid_3 + f_init_1, [f_mid_1, f_mid_2, f_mid_3]



class model_F2I(nn.Module):
    def __init__(self, mid_c, out_c, scale):
        super(model_F2I, self).__init__()

        self.scale = scale

        self.layer_last_1 = nn.Sequential(Conv_(mid_c, out_c, k_size=3, groups=1)
                                         ,nn.LeakyReLU(negative_slope=0.2, inplace=True)
                                         ,Conv_(out_c, out_c, k_size=3, groups=1)
                                         )

    def forward(self, in_x, in_feat):
        f_last_1 = self.layer_last_1(F.interpolate(in_feat
                                                  ,scale_factor=self.scale
                                                  ,mode='bilinear'
                                                  ,align_corners=None
                                                  )
                                    )

        return f_last_1 + F.interpolate(in_x
                                       ,scale_factor=self.scale
                                       ,mode='bilinear'
                                       ,align_corners=None
                                       )



class proposed_model(nn.Module):
    def __init__(self, in_c=3, mid_c=42, out_c=3, scale=4, basic_blocks=None, use_m1=True, use_m2=True):
        super(proposed_model, self).__init__()

        self.layer_I2F = model_I2F(in_c=in_c, mid_c=mid_c, basic_blocks=basic_blocks, use_m1=use_m1, use_m2=use_m2)

        self.layer_F2I = model_F2I(mid_c=mid_c, out_c=out_c, scale=scale)


    def forward(self, in_x):

        mid_feat, list_feats = self.layer_I2F(in_x)

        return [self.layer_F2I(in_x, mid_feat), list_feats]

#===================================================================================
# related to knowledge distillation

class LoGFilter(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super(LoGFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.create_log_kernel(kernel_size, sigma)

    def create_log_kernel(self, kernel_size, sigma):
        ax = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1).float()
        xx, yy = torch.meshgrid(ax, ax)
        xy = xx ** 2 + yy ** 2

        kernel = (xy - 2 * sigma ** 2) / (sigma ** 4) * torch.exp(-xy / (2 * sigma ** 2))
        kernel = kernel - kernel.mean()  # Ensure kernel sum to 0
        kernel = kernel / kernel.abs().sum()  # Normalize to sum to 1
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        return kernel

    def forward(self, x):
        channels = x.shape[1]
        kernel = self.kernel.expand(channels, 1, -1, -1).to(x.device)
        return F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=channels)

class BA_kernel(nn.Module):  # Boundary Aware kernel
    def __init__(self, in_c=3, out_c=3, beta=0.1, kernel_size=5, sigma=1.0):
        self.in_c = in_c
        self.out_c = out_c
        super(BA_kernel, self).__init__()
        self.log_filter = LoGFilter(kernel_size=kernel_size, sigma=sigma)
        self.beta = beta

    def forward(self, x):
        edge = self.log_filter(x)
        return self.beta * x + (1 - self.beta) * edge

class MLP(nn.Module):

    def __init__(self, in_channels, hidden_dim=256, num_layers=3, dropout=0.1):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            *[nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, in_channels)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        x = self.mlp(x)  
        x = self.dropout(x)

        return x

# ===================================================================================
# related to proposed Loss

class proposed_loss(nn.Module):
    def __init__(self, is_amp=True, kd_mode=None, alpha=1.0):
        super(proposed_loss, self).__init__()
        self.loss_l1 = torch.nn.L1Loss()
        self.kd_mode = kd_mode
        self.alpha = alpha

        print("\nproposed_loss")
        print("kd_mode:", self.kd_mode)
        print("alpha:", self.alpha)

    # def kd_origin(in_feat):
        # # original KD
        # return in_feat

    def kd_fitnet(in_feat):
        # FitNet KD
        return in_feat

    def kd_at(self, in_feat):
        # attention transfer (AT)
        eps = 1e-6
        am = torch.pow(torch.abs(in_feat), 2)
        am = torch.sum(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2,3), keepdim=True)
        return torch.div(am, norm+eps)

    def kd_fsp(self, in_feat_1, in_feat_2):
        # flow of solution procedure (FSP)
        _B, _C, _H, _W = in_feat_2.shape
        in_feat_1 = F.adaptive_avg_pool2d(in_feat_1, (_H, _W))
        in_feat_1 = in_feat_1.view(_B, _C, -1)
        in_feat_2 = in_feat_2.view(_B, _C, -1).transpose(1,2)
        return torch.bmm(in_feat_1, in_feat_2) / (_H*_W)

    def kd_fakd(self, in_feat):
        # Feature-affinity based knowledge distillation (FAKD)
        with torch.no_grad(): # from m 1.53 fix 2
            _B, _C, _H, _W = in_feat.shape
            in_feat = in_feat.view(_B, _C, -1)
            norm_fm = in_feat / (torch.sqrt(torch.sum(torch.pow(in_feat,2), 1)).unsqueeze(1).expand(in_feat.shape) + 1e-8)
            sa = norm_fm.transpose(1,2).bmm(norm_fm)
            return sa.unsqueeze(1)

    def kd_ofd(self, fm_s, fm_t):
        margin = calculate_margin(fm_t)
        fm_t = torch.max(fm_t, margin)
        mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
        return (fm_s, fm_t, mask) 

    def kd_mlp(self, in_feat):
        B, C, H, W = in_feat.shape

        in_feat = in_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

        if not hasattr(self, "ch_mlp"):
            self.ch_mlp = MLP(in_channels=C, hidden_dim=256, num_layers=3, dropout=0.1).to(in_feat.device)

        out_feat = self.ch_mlp(in_feat)  

        out_feat = out_feat.view(B, H, W, C).permute(0, 3, 1, 2)

        return out_feat

    def kd_attnfd(self, in_feat):
        eps = 1e-6

        in_feat_norm = in_feat / (torch.norm(in_feat, p=2, dim=(2, 3), keepdim=True) + eps)

        return in_feat_norm


    def kd_bakd(self, in_feat):
        try:
            return self.kd_bakd_conv(in_feat)
        except:
            _B, _C, _H, _W = in_feat.shape
            self.kd_bakd_conv = BA_kernel(in_c=_C, out_c=_C)
            lap_feat = self.kd_bakd_conv(in_feat)
        return lap_feat


    def forward(self, in_pred, in_ans, *args):

        if isinstance(in_pred, list):
            in_pred_sr = in_pred[0]
            in_pred_feat = in_pred[1]
        else:
            in_pred_sr = in_pred
            in_pred_feat = None

        try:
            in_teacher = args[0]
            if isinstance(in_teacher, list):
                in_teacher_sr = in_teacher[0]
                in_teacher_feat = in_teacher[1]
            else:
                in_teacher_sr = in_teacher
                in_teacher_feat = None

        except:
            in_teacher = None

        if in_teacher is None:

            return self.alpha * (self.loss_l1(in_pred_sr, in_ans))
        else:
            _alpha = 0.3

            loss_base = self.loss_l1(in_pred_sr, in_ans)
            loss_kd   = _alpha * self.loss_l1(in_pred_sr,in_teacher_sr)
            loss_feat = 0.0
            if in_teacher_feat is not None:
                _feat_count = len(in_teacher_feat)
            else:
                _feat_count = None

            if self.kd_mode == "kd_origin":
                pass

            else:
                for i_feat in range(_feat_count):
                    if self.kd_mode == "kd_fitnet":
                        s_feat, t_feat = in_pred_feat[i_feat], in_teacher_feat[i_feat]

                    elif self.kd_mode == "kd_at":
                        s_feat, t_feat = self.kd_at(in_pred_feat[i_feat]), self.kd_at(in_teacher_feat[i_feat])

                    elif self.kd_mode == "kd_fsp":
                        s_feat = self.kd_fsp(in_pred_feat[i_feat], in_pred_feat[i_feat + 1])
                        t_feat = self.kd_fsp(in_teacher_feat[i_feat], in_teacher_feat[i_feat + 1])

                    elif self.kd_mode == "kd_fakd":
                        s_feat, t_feat = self.kd_fakd(in_pred_feat[i_feat]), self.kd_fakd(in_teacher_feat[i_feat])

                    elif self.kd_mode == "kd_bakd":
                        s_feat, t_feat = self.kd_bakd(in_pred_feat[i_feat]), self.kd_bakd(in_teacher_feat[i_feat])

                    elif self.kd_mode == "kd_ofd":
                        s_feat, t_feat, mask = self.kd_ofd(in_pred_feat[i_feat], in_teacher_feat[i_feat])
                        loss_feat += _alpha * (self.loss_l1(s_feat, t_feat) * mask).mean()
                        continue

                    elif self.kd_mode == "kd_mlp":
                        s_feat = self.kd_mlp(in_pred_feat[i_feat])
                        t_feat = in_teacher_feat[i_feat]

                    elif self.kd_mode == "kd_attnfd":
                        s_feat, t_feat = self.kd_attnfd(in_pred_feat[i_feat]), self.kd_attnfd(in_teacher_feat[i_feat])

                    elif self.kd_mode ==  "kd_aicsd":
                        s_feat, t_feat = self.kd_aicsd(in_pred_feat[i_feat]), self.kd_aicsd(in_teacher_feat[i_feat])

                    loss_feat += _alpha * self.loss_l1(s_feat, t_feat)

            total_loss = (1 - _alpha) * loss_base + _alpha * (loss_kd + loss_feat / _feat_count)

            if _feat_count is None:
                return self.alpha * ((1 - _alpha) * self.loss_l1(in_pred_sr, in_ans) + _alpha * loss_kd)
            else:
                return loss_base, loss_kd, loss_feat, total_loss



class proposed_loss_ss(nn.Module):
    def __init__(self, is_amp=True, pred_classes=None, ignore_index=-100, alpha=1.0, beta=1.0):
        super(proposed_loss_ss, self).__init__()
        self.eps = 1e-9 # epsilon

        if pred_classes is None:
            _str = "pred_classes must be specified"
            warnings.warn(_str)
            sys.exit(-9)

        self.pred_classes   = pred_classes
        self.ignore_index   = ignore_index
        self.alpha          = alpha
        self.beta           = beta
        print("\nproposed_loss_ss pred_classes is", self.pred_classes)
        print("proposed_loss_ss ignore_index is", self.ignore_index)
        print("proposed_loss_ss alpha and beta is", self.alpha, self.beta)


    def calc_weight(self, in_ans):
        _ans  = in_ans.clone().detach()

        _B, _, _ = _ans.shape
        _ans = _ans.view([-1]) 
        _bin = torch.bincount(_ans, minlength=self.pred_classes)[:self.pred_classes] 
        _norm = 1 - torch.nn.functional.normalize(_bin.float(), p=1, dim=0)

        return _norm

    def forward(self, in_pred, in_ans):

        return torch.nn.functional.cross_entropy(torch.log(in_pred + self.eps), in_ans, ignore_index=self.ignore_index)


print("EOF: model.py")