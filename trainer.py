# trainer.py

_tv = "1.3.3 fix 2"
import copy
import time
import warnings
_str = "\n\n---[ Proposed 2nd paper trainer version: " + str(_tv) + " ]---\n"
warnings.warn(_str)

import os
import numpy as np
import random
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import ignite   

import argparse
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import time
import warnings

from utils.calc_func import *
from utils.data_load_n_save import *
from utils.data_tool import *
from utils.checkpoint import load_checkpoint

from _pkl_mkr.pkl_loader import PklLoader                       

from mps.mp_sssr_plt_saver import plts_saver as plts_saver_sssr   
from mps.mp_dataloader     import DataLoader_multi_worker_FIX       


# Semantic Segmentation
list_ss_model_type_a = ["D3P", "DABNet", "CGNet"]


def accumulate_dict_ious(dict_accumulate, miou, dict_ious):

    if "miou" in dict_accumulate:
        item_current = dict_accumulate["miou"]
    else:
        item_current = (0,0)
        
    dict_accumulate["miou"] = (item_current[0] + 1, item_current[1] + miou)
    
    for i_key in dict_ious:
        if i_key in dict_accumulate:
            item_current = dict_accumulate[i_key]
        else:
            item_current = (0,0)
        
        if dict_ious[i_key] == "NaN":
            item_new = item_current
        else:
            item_new = (item_current[0] + 1, item_current[1] + float(dict_ious[i_key]))
        
        dict_accumulate[i_key] = item_new
    
#=== End of accumulate_dict_ious


def dummy_calc_miou_gray(**kargs):
    int_total_labels  = kargs['int_total_labels']
    in_int_void = kargs['int_void_label']
    
    dict_iou = {}
    
    for i_label in range(int_total_labels):
        if(i_label != in_int_void):
            dict_iou[str(i_label)] = str(-9)

    
    return -9, dict_iou

#=== End of dummy_calc_miou_gray


def dummy_calc_pa_ca_miou_gray(**kargs):
    int_total_labels  = kargs['int_total_labels']
    in_int_void = kargs['int_void_label']
    
    dict_iou = {}
    
    for i_label in range(int_total_labels):
        if(i_label != in_int_void):
            dict_iou[str(i_label)] = str(-9)

    
    return -9, -9, -9, dict_iou

#=== End of dummy_calc_pa_ca_miou_gray

def dummy_calc_pa_ca_miou_gray_tensor(**kargs):
    int_total = kargs['int_total']
    int_void  = kargs['int_void']
    
    dict_iou = {}
    
    for i_label in range(int_total):
        if(i_label != int_void):
            dict_iou[str(i_label)] = str(-9)

    
    return -9, -9, -9, dict_iou

#=== End of dummy_calc_pa_ca_miou_gray_tensor


def one_epoch_gt_kd_sr_2_ss(**kargs):
    FILE_HEAD = "gt_kd_sr_2_ss"

    WILL_SAVE_IMAGE = kargs['WILL_SAVE_IMAGE']

    CALC_WITH_LOGIT           = kargs['CALC_WITH_LOGIT']

    #
    list_mode           = kargs['list_mode']
    i_mode              = kargs['i_mode']
    HP_EPOCH            = kargs['HP_EPOCH']
    i_epoch             = kargs['i_epoch']

    MAX_SAVE_IMAGES     = kargs['MAX_SAVE_IMAGES']
    MUST_SAVE_IMAGE     = kargs['MUST_SAVE_IMAGE']
    BUFFER_SIZE         = kargs['BUFFER_SIZE']
    employ_threshold    = kargs['employ_threshold']

    HP_LABEL_TOTAL          = kargs['HP_LABEL_TOTAL']
    HP_LABEL_VOID           = kargs['HP_LABEL_VOID']
    HP_DATASET_CLASSES      = kargs['HP_DATASET_CLASSES']
    HP_LABEL_ONEHOT_ENCODE  = kargs['HP_LABEL_ONEHOT_ENCODE']
    HP_COLOR_MAP            = kargs['HP_COLOR_MAP']

    PATH_OUT_MODEL      = kargs['PATH_OUT_MODEL']
    PATH_OUT_IMAGE      = kargs['PATH_OUT_IMAGE']
    PATH_OUT_LOG        = kargs['PATH_OUT_LOG']

    device              = kargs['device']

    model_t             = kargs["model_t"]          # Teacher
    optimizer_t         = kargs['optimizer_t']
    criterion_t         = kargs['criterion_t']
    scheduler_t         = kargs['scheduler_t']

    model_s             = kargs['model_s']          # Student
    optimizer_s         = kargs['optimizer_s']
    criterion_s         = kargs['criterion_s']
    scheduler_s         = kargs['scheduler_s']

    model_m            = kargs['model_m']         # seMantic segmentation network
    optimizer_m        = kargs['optimizer_m']     # only SS network
    criterion_m        = kargs['criterion_m']
    scheduler_m        = kargs['scheduler_m']

    HP_MIXUP_A          = kargs['HP_MIXUP_A']       # SR (T, HP_MIXUP_A) and SR (S, 1 - HP_MIXUP_A) mixup value (0 ~ 1)

    amp_scaler          = kargs['amp_scaler']
    ignite_evaluator    = kargs['ignite_evaluator']

    dataloader_input    = kargs['dataloader_input']
    i_batch_max         = len(dataloader_input)

    dict_rb_t          = kargs['dict_rb_t']  # record box SR (T)
    dict_rb_s          = kargs['dict_rb_s']  # record box SR (S)
    dict_rb_m          = kargs['dict_rb_m']  # record box semantic segmentation


    dict_dict_log_total = kargs['dict_dict_log_total']

    dict_dict_log_epoch = {list_mode[0]: {}
                          ,list_mode[1]: {}
                          ,list_mode[2]: {}
                          }

    for i_key in list_mode:
        _str = "batch,file_name,"
        _str += "loss_t_(" + i_key + "),loss_s_(" + i_key + "),loss_m_(" + i_key + "),"
        _str += "PSNR_t_(" + i_key + "),SSIM_t_(" + i_key + "),PSNR_s_(" + i_key + "),SSIM_s_(" + i_key + "),"
        _str += "Pixel_Acc_(" + i_key + "),Class_Acc_(" + i_key + "),mIoU_(" + i_key + "),"
        _str += HP_DATASET_CLASSES

        update_dict_v2("item", _str
                      ,in_dict_dict = dict_dict_log_epoch
                      ,in_dict_key = i_key
                      ,in_print_head = "dict_dict_log_epoch_" + i_key
                      ,is_print = False
                      )

    try:
        torch.cuda.empty_cache()
    except:
        pass

    try:
        timer_gpu_start  = torch.cuda.Event(enable_timing=True)
        timer_gpu_finish = torch.cuda.Event(enable_timing=True)
    except:
        timer_gpu_start  = None
        timer_gpu_finish = None


    if i_mode == "train":
        # sr (t)
        model_t.train()
        optimizer_t.zero_grad()
        print("optimizer_t.zero_grad()")
        # sr (s)
        model_s.train()
        optimizer_s.zero_grad()
        print("optimizer_s.zero_grad()")
        # ss
        model_m.train()
        optimizer_m.zero_grad()
        print("optimizer_m.zero_grad()")
    else:
        model_t.eval()
        model_s.eval()
        model_m.eval()

    i_batch = 0
    timer_cpu_start = time.time()

    try:
        del list_mp_buffer
        list_mp_buffer = []
    except:
        list_mp_buffer = []


    for dataloader_items in dataloader_input:
        dl_str_file_name    = dataloader_items[0]       # (tuple) file name

        if i_mode == "test":
            _bool = bool(set(MUST_SAVE_IMAGE) & set(dl_str_file_name))
        else:
            _bool = False

        if not WILL_SAVE_IMAGE:
            is_pil_needed = False
        else:
            is_pil_needed = (i_batch % (i_batch_max//MAX_SAVE_IMAGES + 1) == 0  or _bool )

        # augm info for train
        dl_str_info_augm    = dataloader_items[3]

        # degraded info for LR images
        try:
            dl_str_info_deg = dataloader_items[9]
            if dl_str_info_deg[0] == False:
                dl_str_info_deg = None
        except:
            dl_str_info_deg = None

        # PIL (HR image & HR label & LR image) -> Full or Patch ver of RAW PIL (no Norm)
        if is_pil_needed:
            dl_pil_img_hr       = tensor_2_list_pils_v1(in_tensor = dataloader_items[1])
            try:
                dl_pil_img_lr   = tensor_2_list_pils_v1(in_tensor = dataloader_items[7])
            except:
                dl_pil_img_lr   = None
        else:
            dl_pil_img_hr = None
            dl_pil_img_lr = None

        try:
            
            dl_ts_lab_hr_gray = torch.round(dataloader_items[4]*255).type(torch.uint8)
        except:
            dl_ts_lab_hr_gray = None

        if is_pil_needed:
            try:
                dl_pil_lab_hr   = tensor_2_list_pils_v1(in_tensor = dataloader_items[4])
            except:
                dl_pil_lab_hr     = None

        # Tensor (HR image & HR label & LR image)
        dl_ts_img_hr        = dataloader_items[2].float()
        try:
            dl_ts_lab_hr    = dataloader_items[5].float()
            dl_ts_lab_hr_void = dataloader_items[6].float()
        except:
            dl_ts_lab_hr    = None
        try:
            dl_ts_img_lr    = dataloader_items[8].float()
        except:
            dl_ts_img_lr    = None


        if timer_gpu_start is not None:
            timer_gpu_start.record()

        dl_ts_img_hr = dl_ts_img_hr.to(device)
        dl_ts_img_lr = dl_ts_img_lr.to(device)


        if dl_ts_lab_hr_gray is not None:
            _dl_ts_lab_hr_gray = dl_ts_lab_hr_gray.clone().detach()
            _dl_ts_lab_hr_gray = torch.squeeze(_dl_ts_lab_hr_gray, dim=1)
            _dl_ts_lab_hr_gray = _dl_ts_lab_hr_gray.type(torch.long)
            _dl_ts_lab_hr_gray = _dl_ts_lab_hr_gray.to(device)
            dl_ts_lab_hr_gray = dl_ts_lab_hr_gray.to(device)

        dl_ts_img_lr_copy = dl_ts_img_lr.clone().detach()   
        dl_ts_img_lr_copy = dl_ts_img_lr_copy.requires_grad_(True)
        dl_ts_img_lr      = dl_ts_img_lr.requires_grad_(True)

        if i_mode == "train":
            # SR (T)
            with torch.no_grad():
                tensor_out_t_raw_copy = model_t(dl_ts_img_lr_copy)

            with torch.cuda.amp.autocast(enabled=True):
                tensor_out_t_raw = model_t(dl_ts_img_lr_copy)
                loss_t = criterion_t(tensor_out_t_raw, dl_ts_img_hr)
                if isinstance(tensor_out_t_raw, list):
                    tensor_out_t = tensor_out_t_raw[0]
                else:
                    tensor_out_t = tensor_out_t_raw

            # SR (S)
            with torch.cuda.amp.autocast(enabled=True):
                tensor_out_s_raw = model_s(dl_ts_img_lr)
                total_loss = criterion_s(tensor_out_s_raw, dl_ts_img_hr, tensor_out_t_raw_copy)
                loss_s = total_loss[-1]

                if isinstance(tensor_out_s_raw, list):
                    tensor_out_s = tensor_out_s_raw[0]
                else:
                    tensor_out_s = tensor_out_s_raw

            with torch.cuda.amp.autocast(enabled=True):
                # Mixup SR (T) and SR (S)
                tensor_out_sr_mixup = tensor_out_t * HP_MIXUP_A + tensor_out_s * (1 - HP_MIXUP_A)
                tensor_out_sr_mixup = tensor_out_sr_mixup.requires_grad_(True)
                tensor_out_seg = model_m(tensor_out_sr_mixup)

                tensor_out_seg_softmax = F.softmax(tensor_out_seg, dim = 1)
                if CALC_WITH_LOGIT:
                    loss_m = criterion_m(tensor_out_seg, _dl_ts_lab_hr_gray)
                else:
                    loss_m = criterion_m(tensor_out_seg_softmax, _dl_ts_lab_hr_gray)

                tensor_out_seg_label = torch.argmax(tensor_out_seg_softmax.clone().detach().requires_grad_(False), dim = 1)

                # combined loss for joint optimization
            combined_loss = loss_t + loss_s + 0.1 * loss_m

            # backward for combined loss
            amp_scaler.scale(combined_loss).backward(retain_graph=False)
            amp_scaler.step(optimizer_t)
            amp_scaler.step(optimizer_s)
            amp_scaler.step(optimizer_m)
            amp_scaler.update()
            optimizer_t.zero_grad()
            optimizer_s.zero_grad()
            optimizer_m.zero_grad()

        else: # val, test
            with torch.no_grad():
                # SR (T)
                tensor_out_t_raw = model_t(dl_ts_img_lr)
                loss_t = criterion_t(tensor_out_t_raw, dl_ts_img_hr)
                if isinstance(tensor_out_t_raw, list):
                    tensor_out_t = tensor_out_t_raw[0]
                else:
                    tensor_out_t = tensor_out_t_raw

                # SR (S)
                tensor_out_s_raw = model_s(dl_ts_img_lr)
                total_loss = criterion_s(tensor_out_s_raw, dl_ts_img_hr, tensor_out_t_raw)
                loss_s = total_loss[-1]

                if isinstance(tensor_out_s_raw, list):
                    tensor_out_s = tensor_out_s_raw[0]
                else:
                    tensor_out_s = tensor_out_s_raw


                if i_batch < 1:
                    print(" for val and test, tensor_out_sr_mixup = tensor_out_s (no mixup to SR-S with SR-T)")
                tensor_out_sr_mixup = tensor_out_s
                tensor_out_seg = model_m(tensor_out_sr_mixup)

                tensor_out_seg_softmax = F.softmax(tensor_out_seg, dim = 1)
                if CALC_WITH_LOGIT:
                    loss_m = criterion_m(tensor_out_seg, _dl_ts_lab_hr_gray)
                else:
                    loss_m = criterion_m(tensor_out_seg_softmax, _dl_ts_lab_hr_gray)

                tensor_out_seg_label = torch.argmax(tensor_out_seg_softmax.clone().detach().requires_grad_(False), dim = 1)


        dict_rb_t[i_mode]["loss"].add_item(loss_t.item())
        dict_rb_s[i_mode]["loss"].add_item(loss_s.item())
        dict_rb_m[i_mode]["loss"].add_item(loss_m.item())


        if timer_gpu_finish is not None:
            timer_gpu_finish.record()

        current_batch_size, _, _, _ = dl_ts_img_hr.shape
        if is_pil_needed:
            list_out_pil_t     = tensor_2_list_pils_v1(in_tensor=tensor_out_t,         is_label=False, is_resized=False)
            list_out_pil_s     = tensor_2_list_pils_v1(in_tensor=tensor_out_s,         is_label=False, is_resized=False)
            list_out_pil_label = tensor_2_list_pils_v1(in_tensor=tensor_out_seg_label, is_label=True,  is_resized=False)

        for i_image in range(current_batch_size):
            with torch.no_grad():
                ignite_in_t  = torch.unsqueeze(tensor_out_t[i_image].to(torch.float32), 0)
                ignite_in_s  = torch.unsqueeze(tensor_out_s[i_image].to(torch.float32), 0)
                ignite_in_hr = torch.unsqueeze(dl_ts_img_hr[i_image], 0)

                ignite_in_t  = ignite_in_t.to(device)
                ignite_in_s  = ignite_in_s.to(device)
                ignite_in_hr = ignite_in_hr.to(device)

                ignite_result = ignite_evaluator.run([[ignite_in_t
                                                      ,ignite_in_hr
                                                    ]])

                out_psnr_t = ignite_result.metrics['psnr']
                out_ssim_t = ignite_result.metrics['ssim']

                dict_rb_t[i_mode]["psnr"].add_item(out_psnr_t)
                dict_rb_t[i_mode]["ssim"].add_item(out_ssim_t)

                ignite_result = ignite_evaluator.run([[ignite_in_s
                                                      ,ignite_in_hr
                                                    ]])

                out_psnr_s = ignite_result.metrics['psnr']
                out_ssim_s = ignite_result.metrics['ssim']

                dict_rb_s[i_mode]["psnr"].add_item(out_psnr_s)
                dict_rb_s[i_mode]["ssim"].add_item(out_ssim_s)

                dict_rb_m[i_mode]["psnr"].add_item(-9.0) # dummy
                dict_rb_m[i_mode]["ssim"].add_item(-9.0) # dummy


            tmp_pa, tmp_ca, tmp_miou, dict_ious = dummy_calc_pa_ca_miou_gray_tensor(ts_ans    = None
                                                                                   ,ts_pred   = None
                                                                                   ,int_total = HP_LABEL_TOTAL
                                                                                   ,int_void  = HP_LABEL_VOID
                                                                                   )

            dict_rb_t[i_mode]["pa"].add_item(tmp_pa)
            dict_rb_t[i_mode]["ca"].add_item(tmp_ca)
            dict_rb_t[i_mode]["ious"].add_item(dict_ious)

            dict_rb_s[i_mode]["pa"].add_item(tmp_pa)
            dict_rb_s[i_mode]["ca"].add_item(tmp_ca)
            dict_rb_s[i_mode]["ious"].add_item(dict_ious)

            # overwrite result from dummy_calc_pa_ca_miou_gray_tensor
            tmp_pa, tmp_ca, tmp_miou, dict_ious = calc_pa_ca_miou_gray_tensor(ts_ans    = dl_ts_lab_hr_gray[i_image][0]
                                                                             ,ts_pred   = tensor_out_seg_label[i_image]
                                                                             ,int_total = HP_LABEL_TOTAL
                                                                             ,int_void  = HP_LABEL_VOID
                                                                             ,device    = device
                                                                             )

            dict_rb_m[i_mode]["pa"].add_item(tmp_pa)
            dict_rb_m[i_mode]["ca"].add_item(tmp_ca)
            dict_rb_m[i_mode]["ious"].add_item(dict_ious)

            tmp_ious = ""
            for i_key in dict_ious:
                tmp_ious += "," + dict_ious[i_key]

            tmp_str_contents = (
                    str(i_batch + 1) + "," + dl_str_file_name[i_image] +
                    "," + str(loss_t.item()) +
                    "," + str(loss_s.item()) +
                    "," + str(loss_m.item()) +
                    "," + str(out_psnr_t) +
                    "," + str(out_ssim_t) +
                    "," + str(out_psnr_s) +
                    "," + str(out_ssim_s) +
                    "," + str(tmp_pa) +
                    "," + str(tmp_ca) +
                    "," + str(tmp_miou) +
                    tmp_ious
            )

            update_dict_v2("", tmp_str_contents
                          ,in_dict_dict = dict_dict_log_epoch
                          ,in_dict_key = i_mode
                          ,is_print = False
                          )

            if is_pil_needed and i_image < 2:
                if len(list_mp_buffer) >= BUFFER_SIZE:
                    plts_saver_sssr(list_mp_buffer, is_best=(i_mode=="test"), no_employ=(employ_threshold >= len(list_mp_buffer)))
                    try:
                        del list_mp_buffer
                        list_mp_buffer = []
                    except:
                        list_mp_buffer = []

                # epoch 마다 n 배치 정도의 결과 이미지를 저장해봄
                plt_title = "File name: " + dl_str_file_name[i_image]
                plt_title += "\n" + dl_str_info_augm[i_image]
                try:
                    # 현재 patch의 degrad- 옵션 불러오기
                    plt_title += "\n" + dl_str_info_deg[i_image]
                except:
                    pass
                plt_title += "\nPSNR_t: " + str(round(out_psnr_t, 4)) + "  SSIM_t: " + str(round(out_ssim_t, 4))
                plt_title += "\nPSNR_s: " + str(round(out_psnr_s, 4)) + "  SSIM_s: " + str(round(out_ssim_s, 4))
                plt_title += "\nPA: "   + str(round(tmp_pa, 4))   + "  CA: " + str(round(tmp_ca, 4))
                plt_title += "\nmIoU: " + str(round(tmp_miou, 4))

                tmp_file_name = (i_mode + "_" + str(i_epoch + 1) + "_" + str(i_batch + 1) + "_"
                                +dl_str_file_name[i_image]
                                )

                list_mp_buffer.append((# 0 (model name)
                                       "SSSR_D"
                                       # 1 ~ 6 (pils)
                                      ,dl_pil_img_hr[i_image], label_2_RGB(dl_pil_lab_hr[i_image], HP_COLOR_MAP)
                                      ,label_2_RGB(list_out_pil_label[i_image], HP_COLOR_MAP), dl_pil_img_lr[i_image]
                                      ,list_out_pil_t[i_image], list_out_pil_s[i_image]
                                       # 7 ~ 12 (sub title)
                                      ,"HR Image", "Label Answer", "Predicted", "LR Image", "SR (t) Image", "SR (s) Image"
                                       # 13 (path plt)
                                      ,PATH_OUT_IMAGE + i_mode + "/" + FILE_HEAD  +"_" + str(i_epoch + 1)
                                       # 14 ~ 17 (data CF)
                                      ,None, None, None, None
                                       # 18 (path CF)
                                      ,PATH_OUT_IMAGE + i_mode + "/" + str(i_epoch + 1)
                                       # 19 (path pil)
                                      ,PATH_OUT_IMAGE + i_mode + "/_Images/" + FILE_HEAD + "_" + str(i_epoch + 1)
                                       # 20 (plt title)
                                      ,plt_title
                                       # 21 (file name)
                                      ,tmp_file_name
                                      )
                                     )

        for i_key in ["loss", "psnr", "ssim", "pa", "ca", "ious"]:
            dict_rb_t[i_mode][i_key].update_batch()
            dict_rb_s[i_mode][i_key].update_batch()
            dict_rb_m[i_mode][i_key].update_batch()

        try:
            # GPU timer record
            torch.cuda.synchronize()
            timer_gpu_record = str(round(timer_gpu_start.elapsed_time(timer_gpu_finish) / 1000, 4))
        except:
            timer_gpu_record = "FAIL"

        timer_cpu_record = str(round(time.time() - timer_cpu_start, 4))

        print("\rin", i_mode, (i_epoch + 1), "/", HP_EPOCH, "-", (i_batch + 1), "/", i_batch_max, "-"
             ,"CPU:", timer_cpu_record, " GPU:", timer_gpu_record
             ,end = '')

        timer_cpu_start = time.time()
    

        i_batch += 1

    print("")

    if list_mp_buffer:
        plts_saver_sssr(list_mp_buffer, is_best=(i_mode=="test"), no_employ=(employ_threshold >= len(list_mp_buffer)))


    tmp_str_contents  =       str(dict_rb_t[i_mode]["loss"].update_epoch(is_return = True, path = PATH_OUT_LOG, is_print_sub = True))
    tmp_str_contents += "," + str(dict_rb_s[i_mode]["loss"].update_epoch(is_return = True, path = PATH_OUT_LOG, is_print_sub = True))
    tmp_str_contents += "," + str(dict_rb_m[i_mode]["loss"].update_epoch(is_return = True, path = PATH_OUT_LOG, is_print_sub = True))


    for i_key in ["psnr", "ssim"]:
        tmp_str_contents += "," + str(dict_rb_t[i_mode][i_key].update_epoch(is_return = True, path = PATH_OUT_LOG,is_print_sub = True ,is_update_graph = True))

    for i_key in ["psnr", "ssim"]:
        tmp_str_contents += "," + str(dict_rb_s[i_mode][i_key].update_epoch(is_return = True, path = PATH_OUT_LOG,is_print_sub = True ,is_update_graph = True))

    for i_key in ["pa", "ca", "ious"]:
        dict_rb_t[i_mode][i_key].update_epoch(is_return = False, path = PATH_OUT_LOG, is_print_sub = False ,is_update_graph = False)
        dict_rb_s[i_mode][i_key].update_epoch(is_return = False, path = PATH_OUT_LOG, is_print_sub = False ,is_update_graph = False)

    for i_key in ["psnr", "ssim"]:
        dict_rb_m[i_mode][i_key].update_epoch(is_return = False, path = PATH_OUT_LOG ,is_print_sub = False ,is_update_graph = False)

    for i_key in ["pa", "ca", "ious"]:
        tmp_str_contents += "," + str(dict_rb_m[i_mode][i_key].update_epoch(is_return = True, path = PATH_OUT_LOG,is_print_sub = True ,is_update_graph = True))

    update_dict_v2(str(i_epoch + 1), tmp_str_contents
                  ,in_dict_dict = dict_dict_log_total
                  ,in_dict_key = i_mode
                  ,in_print_head = "dict_dict_log_total_" + i_mode
                  ,is_print = False
                  )

    if i_mode == "train":
        # SR (T)
        dict_rb_t[i_mode]["lr"].add_item(optimizer_t.param_groups[0]['lr'])
        dict_rb_t[i_mode]["lr"].update_batch()
        # scheduler
        scheduler_t.step()
        print("scheduler_t.step()")
        dict_rb_t[i_mode]["lr"].update_epoch(is_return = False, path = PATH_OUT_LOG, is_print_sub = True)

        # SR (S)
        dict_rb_s[i_mode]["lr"].add_item(optimizer_s.param_groups[0]['lr'])
        dict_rb_s[i_mode]["lr"].update_batch()
        # scheduler
        scheduler_s.step()
        print("scheduler_s.step()")
        dict_rb_s[i_mode]["lr"].update_epoch(is_return = False, path = PATH_OUT_LOG, is_print_sub = True)

        # semantic segmentation
        dict_rb_m[i_mode]["lr"].add_item(optimizer_m.param_groups[0]['lr'])
        dict_rb_m[i_mode]["lr"].update_batch()
        # scheduler
        scheduler_m.step()
        print("scheduler_m.step()")
        dict_rb_m[i_mode]["lr"].update_epoch(is_return = False, path = PATH_OUT_LOG, is_print_sub = True)

    dict_2_txt_v2(in_file_path = PATH_OUT_LOG + i_mode + "/"
                 ,in_file_name = FILE_HEAD + "_log_epoch_" + i_mode + "_" + str(i_epoch + 1) + ".csv"
                 ,in_dict_dict = dict_dict_log_epoch
                 ,in_dict_key = i_mode
                 )

    dict_2_txt_v2(in_file_path = PATH_OUT_LOG
                 ,in_file_name = FILE_HEAD + "_log_total_" + i_mode + ".csv"
                 ,in_dict_dict = dict_dict_log_total
                 ,in_dict_key = i_mode
                 )

    save_interval = 100

    if i_mode == "train" and (i_epoch + 1) % save_interval == 0:
        tmp_path = PATH_OUT_MODEL + "check_points/"
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        print("\n[--- checkpoint", str(i_epoch + 1), "saved ---]\n")
        torch.save({'epoch': (i_epoch + 1)
                   ,'model_t_state_dict'    : model_t.state_dict()      # (state_dict) model_t.state_dict()
                   ,'model_s_state_dict'    : model_s.state_dict()      # (state_dict) model_s.state_dict()
                   ,'model_m_state_dict'    : model_m.state_dict()      # (state_dict) model_m.state_dict()
                   ,'optimizer_t_state_dict': optimizer_t.state_dict()  # (state_dict) optimizer_t.state_dict()
                   ,'optimizer_s_state_dict': optimizer_s.state_dict()  # (state_dict) optimizer_s.state_dict()
                   ,'optimizer_m_state_dict': optimizer_m.state_dict()  # (state_dict) optimizer_m.state_dict()
                   ,'scheduler_t_state_dict': scheduler_t.state_dict()  # (state_dict) scheduler_t.state_dict()
                   ,'scheduler_s_state_dict': scheduler_s.state_dict()  # (state_dict) scheduler_s.state_dict()
                   ,'scheduler_m_state_dict': scheduler_m.state_dict()  # (state_dict) scheduler_m.state_dict()
                   }
                  ,tmp_path + str(i_epoch + 1) + "_" + FILE_HEAD + "_check_point.tar"
                  )

    if i_mode == "val":
        tmp_is_best = dict_rb_t[i_mode]["psnr"].is_best_max or dict_rb_t[i_mode]["ssim"].is_best_max or dict_rb_s[i_mode]["ssim"].is_best_max or dict_rb_s[i_mode]["psnr"].is_best_max or dict_rb_m[i_mode]["ious"].is_best_max # PSNR 기준 or mIoU 기준

        if tmp_is_best:
            print("\n< Best Valid Epoch > Model State Dict saved")
            tmp_path = PATH_OUT_MODEL + "state_dicts/" 
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)
            # Teacher, Student, seMantic segmentation model state dict
            torch.save(model_t.state_dict(), tmp_path + FILE_HEAD + "_" + str(i_epoch + 1) +'_t_msd.pt')
            torch.save(model_s.state_dict(), tmp_path + FILE_HEAD + "_" + str(i_epoch + 1) +'_s_msd.pt')
            torch.save(model_m.state_dict(), tmp_path + FILE_HEAD + "_" + str(i_epoch + 1) +'_m_msd.pt')

        else:
            print("\n< Not a Best Valid Epoch >")


    else:
        tmp_is_best = None

    return tmp_is_best


#=== End of one_epoch_gt_kd_sr_2_ss


def trainer_(**kargs):
    dict_log_init = kargs['dict_log_init']

    try:
        WILL_SAVE_IMAGE = bool(kargs['WILL_SAVE_IMAGE'])
    except:
        WILL_SAVE_IMAGE = True

    try:
        SKIP_TEST_UNTIL = int (kargs['SKIP_TEST_UNTIL'])
    except:
        SKIP_TEST_UNTIL = 0

    update_dict_v2("", ""
                  ,"", "For short training time"
                  ,"", "WILL_SAVE_IMAGE: " + str(WILL_SAVE_IMAGE)
                  ,"", "SKIP_TEST_UNTIL: " + str(SKIP_TEST_UNTIL)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )


    CALC_WITH_LOGIT = bool(kargs['CALC_WITH_LOGIT'])
    if CALC_WITH_LOGIT:
        _str = "use logit when calculate semantic segmentation loss"
    else:
        _str = "use softmax when calculate semantic segmentation loss"
    update_dict_v2("", ""
                  ,"", "CALC_WITH_LOGIT: " + str(CALC_WITH_LOGIT)
                  ,"", _str
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )
    warnings.warn(_str)

    try:
        HP_DATASET_NAME = str(kargs['HP_DATASET_NAME'])
    except:
        HP_DATASET_NAME = "No Info"

    try:
        HP_LABEL_TOTAL  = kargs['HP_LABEL_TOTAL']
    except:
        HP_LABEL_TOTAL  = 2

    try:
        HP_LABEL_VOID   = kargs['HP_LABEL_VOID']
    except:
        HP_LABEL_VOID   = None

    if HP_LABEL_VOID is None:
        HP_LABEL_VOID   = HP_LABEL_TOTAL
        _str = "Void Label no exist"
        warnings.warn(_str)

    try:
        HP_DATASET_CLASSES = kargs['HP_DATASET_CLASSES']
    except:
        HP_DATASET_CLASSES = None

    if HP_DATASET_CLASSES == None:
        HP_DATASET_CLASSES = ""
        if HP_LABEL_TOTAL > HP_LABEL_VOID:
            _dummy_class = HP_LABEL_TOTAL - 1
        else:
            _dummy_class = HP_LABEL_TOTAL

        for i_class in range(_dummy_class):
            if i_class == 0:
                HP_DATASET_CLASSES += str(i_class)
            else:
                HP_DATASET_CLASSES += "," + str(i_class)

        _str = "no class info"
        warnings.warn(_str)


    MUST_SAVE_IMAGE = kargs['MUST_SAVE_IMAGE']

    try:
        MAX_SAVE_IMAGES = int(kargs['MAX_SAVE_IMAGES'])
        if MAX_SAVE_IMAGES < 1:
            MAX_SAVE_IMAGES = 1
            print("MAX_SAVE_IMAGES should be >= 1. It fixed to", MAX_SAVE_IMAGES)
    except:
        MAX_SAVE_IMAGES = 10

    employ_threshold = 7

    try:
        BUFFER_SIZE = kargs['BUFFER_SIZE']
        if BUFFER_SIZE < 1:
            print("BUFFER_SIZE should be > 0")
            BUFFER_SIZE = 60
    except:
        BUFFER_SIZE = 60

    print("BUFFER_SIZE set to", BUFFER_SIZE)
    print("BUFFER_SIZE set to", BUFFER_SIZE)



    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_cpu = torch.device('cpu')

    HP_SEED = kargs['HP_SEED']
    random.seed(HP_SEED)
    np.random.seed(HP_SEED)
    torch.manual_seed(HP_SEED)

    update_dict_v2("", ""
                  ,"", "random numpy pytorch"
                  ,"", "HP_SEED: " + str(HP_SEED)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )

    if device == torch.device('cuda'):
        _str = "RUN with cuda"
        warnings.warn(_str)
        torch.cuda.manual_seed(HP_SEED)
        torch.cuda.manual_seed_all(HP_SEED)
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    else:
        _str = "RUN on CPU"
        warnings.warn(_str)

    HP_EPOCH        = kargs['HP_EPOCH']
    HP_BATCH_TRAIN  = kargs['HP_BATCH_TRAIN']

    try:
        HP_VALID_WITH_PATCH = kargs['HP_VALID_WITH_PATCH']  # (bool)
    except:
        HP_VALID_WITH_PATCH = True

    HP_BATCH_VAL    = kargs['HP_BATCH_VAL']
    HP_BATCH_TEST   = kargs['HP_BATCH_TEST']

    try:
        HP_NUM_WORKERS = int(kargs['HP_NUM_WORKERS'])
        _total_worker = mp.cpu_count()
        if _total_worker <= 2:
            print("total workers are not enough to use multi-worker")
            HP_NUM_WORKERS = 0
        elif _total_worker < HP_NUM_WORKERS*2:
            print("too much worker!")
            HP_NUM_WORKERS = int(_total_worker//2)
    except:
        HP_NUM_WORKERS = 0

    update_dict_v2("", ""
                  ,"", "epoch: " + str(HP_EPOCH)
                  ,"", "batch"
                  ,"", "HP_BATCH_TRAIN: " + str(HP_BATCH_TRAIN)
                  ,"", "HP_BATCH_VAL:   " + str(HP_BATCH_VAL)
                  ,"", "HP_BATCH_TEST:  " + str(HP_BATCH_TEST)
                  ,"", "HP_NUM_WORKERS for train: " + str(HP_NUM_WORKERS)
                  ,"", "Gradient Accumulation no use"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )

    # path_input
    PATH_BASE_IN        = kargs['PATH_BASE_IN']
    NAME_FOLDER_TRAIN   = kargs['NAME_FOLDER_TRAIN']
    NAME_FOLDER_VAL     = kargs['NAME_FOLDER_VAL']
    NAME_FOLDER_TEST    = kargs['NAME_FOLDER_TEST']
    NAME_FOLDER_IMAGES  = kargs['NAME_FOLDER_IMAGES']
    NAME_FOLDER_LABELS  = kargs['NAME_FOLDER_LABELS']

    # path_output
    PATH_OUT_IMAGE = kargs['PATH_OUT_IMAGE']
    if PATH_OUT_IMAGE[-1] != '/':
        PATH_OUT_IMAGE += '/'
    PATH_OUT_MODEL = kargs['PATH_OUT_MODEL']
    if PATH_OUT_MODEL[-1] != '/':
        PATH_OUT_MODEL += '/'
    PATH_OUT_LOG = kargs['PATH_OUT_LOG']
    if PATH_OUT_LOG[-1] != '/':
        PATH_OUT_LOG += '/'

    try:
        is_force_fix = kargs['is_force_fix']
        _w, _h = kargs['force_fix_size_hr']
        force_fix_size_hr = (int(_w), int(_h))

        update_dict_v2("", "Train & Valid image Margin -> patch"
                      ,"", "Margin include(W H): (" + str(int(_w)) + " " + str(int(_h)) + ")"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )

    except:
        is_force_fix = False
        force_fix_size_hr = None
        update_dict_v2("", "Train & Valid image Margin -> patch x"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )


    model_t         = kargs['model_t']          # SR (Teacher)
    optimizer_t     = kargs['optimizer_t']
    criterion_t     = kargs['criterion_t']      # loss function
    scheduler_t     = kargs['scheduler_t']

    model_s         = kargs['model_s']          # SR (Student)
    optimizer_s     = kargs['optimizer_s']
    criterion_s     = kargs['criterion_s']
    scheduler_s     = kargs['scheduler_s']

    model_m         = kargs['model_m']
    optimizer_m     = kargs['optimizer_m']
    criterion_m     = kargs['criterion_m']
    scheduler_m     = kargs['scheduler_m']


    HP_MIXUP_A = float(kargs['HP_MIXUP_A']) 

    update_dict_v2("", ""
                  ,"", "(gt_kd_sr_2_ss, HP_MIXUP_A) SR (T) and SR (S) mixup: " + str(HP_MIXUP_A)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )

    TRAINER_MODE = "SSSR"

    print("\n#=========================#")
    print("  Trainer Mode:", TRAINER_MODE)
    print("#=========================#\n")

    try:
        HP_DETECT_LOSS_ANOMALY      = False
    except:
        HP_DETECT_LOSS_ANOMALY      = False


    if HP_DETECT_LOSS_ANOMALY:
        update_dict_v2("", ""
                      ,"", "Train loss -> detect_anomaly o"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    else:
        update_dict_v2("", ""
                      ,"", "Train loss -> detect_anomaly x"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )

    amp_scaler = torch.cuda.amp.GradScaler(enabled = True)
    update_dict_v2("", ""
                  ,"", "Automatic Mixed Precision o"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )

    # [data augmentation]--------------------
    try:
        HP_AUGM_LITE            = kargs['HP_AUGM_LITE']
        HP_AUGM_LITE_FLIP_HORI  = bool(kargs['HP_AUGM_LITE_FLIP_HORI'])
        HP_AUGM_LITE_FLIP_VERT  = bool(kargs['HP_AUGM_LITE_FLIP_VERT'])

    except:
        HP_AUGM_LITE            = False
        HP_AUGM_LITE_FLIP_HORI  = False
        HP_AUGM_LITE_FLIP_VERT  = False

    if not HP_AUGM_LITE:
        HP_AUGM_RANGE_CROP_INIT = kargs['HP_AUGM_RANGE_CROP_INIT']
        HP_AUGM_ROTATION_MAX    = kargs['HP_AUGM_ROTATION_MAX']
        HP_AUGM_PROB_FLIP       = kargs['HP_AUGM_PROB_FLIP']
        HP_AUGM_PROB_CROP       = kargs['HP_AUGM_PROB_CROP']
        HP_AUGM_PROB_ROTATE     = kargs['HP_AUGM_PROB_ROTATE']

        # colorJitter
        HP_CJ_BRIGHTNESS        = kargs['HP_CJ_BRIGHTNESS']
        HP_CJ_CONTRAST          = kargs['HP_CJ_CONTRAST']
        HP_CJ_SATURATION        = kargs['HP_CJ_SATURATION']
        HP_CJ_HUE               = kargs['HP_CJ_HUE']

    else:
        update_dict_v2("", ""
                  ,"", "Augmentation LITE mode applied"
                  ,"", "FLIP Horizontal: " + str(HP_AUGM_LITE_FLIP_HORI)
                  ,"", "FLIP Vertical: " + str(HP_AUGM_LITE_FLIP_VERT)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )

        HP_AUGM_RANGE_CROP_INIT = None
        HP_AUGM_ROTATION_MAX    = None
        HP_AUGM_PROB_FLIP       = None
        HP_AUGM_PROB_CROP       = None
        HP_AUGM_PROB_ROTATE     = None
        HP_CJ_BRIGHTNESS        = [1,1]
        HP_CJ_CONTRAST          = [1,1]
        HP_CJ_SATURATION        = [1,1]
        HP_CJ_HUE               = [0,0]

    update_dict_v2("", ""
                  ,"", "Augmentation"
                  ,"", "HP_AUGM_LITE:             " + str(HP_AUGM_LITE)
                  ,"", "HP_AUGM_LITE_FLIP_HORI:   " + str(HP_AUGM_LITE_FLIP_HORI)
                  ,"", "HP_AUGM_LITE_FLIP_VERT:   " + str(HP_AUGM_LITE_FLIP_VERT)
                  ,"", "HP_AUGM_RANGE_CROP_INIT:  " + "".join(str(HP_AUGM_RANGE_CROP_INIT).split(","))
                  ,"", "HP_AUGM_ROTATION_MAX:     " + str(HP_AUGM_ROTATION_MAX)
                  ,"", "HP_AUGM_PROB_FLIP:        " + str(HP_AUGM_PROB_FLIP)
                  ,"", "HP_AUGM_PROB_CROP:        " + str(HP_AUGM_PROB_CROP)
                  ,"", "HP_AUGM_PROB_ROTATE:      " + str(HP_AUGM_PROB_ROTATE)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )

    update_dict_v2("", ""
                  ,"", "ColorJitter"
                  ,"", "brightness: ( " + " ".join([str(t_element) for t_element in HP_CJ_BRIGHTNESS]) +" )"
                  ,"", "contrast:   ( " + " ".join([str(t_element) for t_element in HP_CJ_CONTRAST])   +" )"
                  ,"", "saturation: ( " + " ".join([str(t_element) for t_element in HP_CJ_SATURATION]) +" )"
                  ,"", "hue:        ( " + " ".join([str(t_element) for t_element in HP_CJ_HUE])        +" )"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )

    try:
        HP_AUGM_RANDOM_SCALER = kargs['HP_AUGM_RANDOM_SCALER']
    except:
        HP_AUGM_RANDOM_SCALER = [1.0]

    update_dict_v2("", ""
                  ,"", "RANDOM_SCALER"
                  ,"", "List: [ " + " ".join([str(t_element) for t_element in HP_AUGM_RANDOM_SCALER]) +" ]"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )

    try:
        is_norm_in_transform_to_tensor = kargs['is_norm_in_transform_to_tensor']
    except:
        is_norm_in_transform_to_tensor = False

    if is_norm_in_transform_to_tensor:
        HP_TS_NORM_MEAN = kargs['HP_TS_NORM_MEAN']
        HP_TS_NORM_STD = kargs['HP_TS_NORM_STD']
        transform_to_ts_img = transforms.Compose([transforms.ToTensor()
                                                 ,transforms.Normalize(mean = HP_TS_NORM_MEAN, std = HP_TS_NORM_STD)
                                                 ,
                                                 ])

        transform_ts_inv_norm = transforms.Compose([
                                                    transforms.Normalize(mean = [ 0., 0., 0. ]
                                                                        ,std = [ 1/HP_TS_NORM_STD[0], 1/HP_TS_NORM_STD[1], 1/HP_TS_NORM_STD[2] ])

                                                   ,transforms.Normalize(mean = [ -HP_TS_NORM_MEAN[0], -HP_TS_NORM_MEAN[1], -HP_TS_NORM_MEAN[2] ]
                                                                        ,std = [ 1., 1., 1. ])

                                                   ,
                                                   ])

        update_dict_v2("", ""
                      ,"", "in_x normalized"
                      ,"", "mean=[ " + str(HP_TS_NORM_MEAN[0]) + " " + str(HP_TS_NORM_MEAN[1]) + " "+ str(HP_TS_NORM_MEAN[2]) + " ]"
                      ,"", "std=[ " + str(HP_TS_NORM_STD[0]) + " " + str(HP_TS_NORM_STD[1]) + " "+ str(HP_TS_NORM_STD[2]) + " ]"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )
    else:
        transform_to_ts_img = transforms.Compose([
                                                  transforms.ToTensor()
                                                 ])

        update_dict_v2("", ""
                      ,"", "in_x normalized x"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )

    option_apply_degradation = True

    if option_apply_degradation:

        try:
            PATH_BASE_IN_SUB = kargs['PATH_BASE_IN_SUB']
            HP_DG_CSV_NAME = kargs['HP_DG_CSV_NAME']
            if not PATH_BASE_IN_SUB[-1] == "/":
                PATH_BASE_IN_SUB += "/"

            dict_loaded_pils = load_pils_2_dict(in_path = PATH_BASE_IN_SUB

                                               ,in_path_sub = NAME_FOLDER_IMAGES
                                               )
            print("Pre-Degraded images loaded from:", PATH_BASE_IN_SUB + NAME_FOLDER_IMAGES)

            dict_dg_csv = csv_2_dict(path_csv = PATH_BASE_IN_SUB + HP_DG_CSV_NAME)
            print("Pre-Degrade option csv re-loaded from:", PATH_BASE_IN_SUB + HP_DG_CSV_NAME)

            flag_pre_degraded_images_loaded = True
            tmp_log_pre_degraded_images_load = "Degraded image loaded"
        except:
            print("(exc) Pre-Degraded images load FAIL")
            flag_pre_degraded_images_loaded = False
            tmp_log_pre_degraded_images_load = "Degraded image not loaded"

        update_dict_v2("", ""
                      ,"", "Degraded option: " + tmp_log_pre_degraded_images_load
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )

        # scale_factor
        HP_DG_SCALE_FACTOR = kargs['HP_DG_SCALE_FACTOR']

        update_dict_v2("", ""
                      ,"", "Degradation"
                      ,"", "DG file path: " + PATH_BASE_IN_SUB + HP_DG_CSV_NAME
                      ,"", "Scale Factor  = x" + str(HP_DG_SCALE_FACTOR)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )



    else:
        update_dict_v2("", ""
                      ,"", "Degradation"
                      ,"",  "Train & Valid & Test : No Degradation"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )


    # [data & model load]--------------------------
    PATH_ALTER_HR_IMAGE = None

    if TRAINER_MODE == "SS" or TRAINER_MODE == "SSSR":
        tmp_is_return_label = True
        if HP_AUGM_LITE:
            sys.exit(_str)
        save_graph_ss = True
    else:
        save_graph_ss = False

    if TRAINER_MODE == "SR" or TRAINER_MODE == "SSSR":
        tmp_is_return_image_lr = True
        save_graph_sr = True
    else:
        save_graph_sr = False

    if tmp_is_return_label:
        HP_COLOR_MAP    = kargs['HP_COLOR_MAP']

        update_dict_v2("", ""
                      ,"", "dataset: " + HP_DATASET_NAME
                      ,"", "RGB mapping"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )

        for i_key in HP_COLOR_MAP:
            _color_map = ""
            for i_color in HP_COLOR_MAP[i_key]:
                _color_map += " " + str(i_color)

            update_dict_v2("", str(i_key) + ":" + _color_map
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )

        if HP_LABEL_VOID == HP_LABEL_TOTAL:
            update_dict_v2("", ""
                          ,"", "number of label: " + str(HP_LABEL_TOTAL)
                          ,"", "no void label"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        else:
            update_dict_v2("", ""
                          ,"", "number of label(include void): " + str(HP_LABEL_TOTAL)
                          ,"", "void label number: " + str(HP_LABEL_VOID)
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )

        try:
            HP_LABEL_ONEHOT_ENCODE = kargs['HP_LABEL_ONEHOT_ENCODE']
        except:
            HP_LABEL_ONEHOT_ENCODE = False

        if HP_LABEL_ONEHOT_ENCODE:
            _str = "Label one-hot encode"
            warnings.warn(_str)
            update_dict_v2("", "Label one-hot encode"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        else:
            _str = "No Label one-hot encode"
            warnings.warn(_str)
            update_dict_v2("", "No Label one-hot encode"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )

        try:
            HP_LABEL_DILATED = kargs['HP_LABEL_DILATED']
        except:
            HP_LABEL_DILATED = False

        if HP_LABEL_DILATED:
            _str = "DILATION for Labels in train: Activated"
            warnings.warn(_str)
            update_dict_v2("", ""
                          ,"", "Label Dilation"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )
        else:
            _str = "DILATION for Labels in train: Deactivated"
            warnings.warn(_str)
            update_dict_v2("", ""
                          ,"", "No Label Dilation"
                          ,in_dict = dict_log_init
                          ,in_print_head = "dict_log_init"
                          )

        if HP_LABEL_DILATED and not HP_LABEL_ONEHOT_ENCODE:
            _str = "If label is dilated, it must be in one-hot form."
            sys.exit(_str)

        if is_force_fix:
            try:
                HP_LABEL_VERIFY             = kargs['HP_LABEL_VERIFY']
                HP_LABEL_VERIFY_TRY_CEILING = kargs['HP_LABEL_VERIFY_TRY_CEILING']
                HP_LABEL_VERIFY_CLASS_MIN   = kargs['HP_LABEL_VERIFY_CLASS_MIN']
                HP_LABEL_VERIFY_RATIO_MAX   = kargs['HP_LABEL_VERIFY_RATIO_MAX']
            except:
                HP_LABEL_VERIFY             = False
                HP_LABEL_VERIFY_TRY_CEILING = None
                HP_LABEL_VERIFY_CLASS_MIN   = None
                HP_LABEL_VERIFY_RATIO_MAX   = None

            if HP_LABEL_VERIFY:
                update_dict_v2("", ""
                              ,"", "Label Verify in train"
                              ,"", "re-crop max " + str(HP_LABEL_VERIFY_TRY_CEILING)
                              ,"", "class min: " + str(HP_LABEL_VERIFY_CLASS_MIN)
                              ,"", "class max ratio (0 ~ 1): " + str(HP_LABEL_VERIFY_RATIO_MAX)
                              ,in_dict = dict_log_init
                              ,in_print_head = "dict_log_init"
                              )
            else:
                update_dict_v2("", ""
                              ,"", "Label Verify in train x"
                              ,in_dict = dict_log_init
                              ,in_print_head = "dict_log_init"
                              )


    else:
        HP_COLOR_MAP                = None
        HP_LABEL_ONEHOT_ENCODE      = False
        HP_LABEL_DILATED            = False
        HP_LABEL_VERIFY             = False
        HP_LABEL_VERIFY_TRY_CEILING = None
        HP_LABEL_VERIFY_CLASS_MIN   = None
        HP_LABEL_VERIFY_RATIO_MAX   = None


    dataset_train = Custom_Dataset_V6(# Return: dict key order -> 'file_name'
                                      #                         , 'pil_img_hr', 'ts_img_hr', 'info_augm'
                                      #                         , 'pil_lab_hr', 'ts_lab_hr'
                                      #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                      name_memo                     = 'train'
                                     ,in_path_dataset               = PATH_BASE_IN
                                     ,in_category                   = NAME_FOLDER_TRAIN
                                     ,in_name_folder_image          = NAME_FOLDER_IMAGES

                                      #--- options for train
                                     ,is_train                      = True
                                      # below options can be skipped when above option is False
                                     ,opt_augm_lite                 = HP_AUGM_LITE
                                     ,opt_augm_lite_flip_hori       = HP_AUGM_LITE_FLIP_HORI
                                     ,opt_augm_lite_flip_vert       = HP_AUGM_LITE_FLIP_VERT
                                     ,opt_augm_crop_init_range      = HP_AUGM_RANGE_CROP_INIT
                                     ,opt_augm_rotate_max_degree    = HP_AUGM_ROTATION_MAX
                                     ,opt_augm_prob_flip            = HP_AUGM_PROB_FLIP
                                     ,opt_augm_prob_crop            = HP_AUGM_PROB_CROP
                                     ,opt_augm_prob_rotate          = HP_AUGM_PROB_ROTATE
                                     ,opt_augm_cj_brigntess         = HP_CJ_BRIGHTNESS
                                     ,opt_augm_cj_contrast          = HP_CJ_CONTRAST
                                     ,opt_augm_cj_saturation        = HP_CJ_SATURATION
                                     ,opt_augm_cj_hue               = HP_CJ_HUE
                                     ,opt_augm_random_scaler        = HP_AUGM_RANDOM_SCALER

                                      #--- options for HR image
                                     ,in_path_alter_hr_image        = PATH_ALTER_HR_IMAGE

                                      #--- options for HR label
                                     ,is_return_label               = tmp_is_return_label
                                      # below options can be skipped when above option is False
                                     ,in_name_folder_label          = NAME_FOLDER_LABELS
                                     ,label_number_total            = HP_LABEL_TOTAL
                                     ,label_number_void             = HP_LABEL_VOID
                                     ,is_label_dilated              = HP_LABEL_DILATED

                                     ,is_label_onehot_encode        = HP_LABEL_ONEHOT_ENCODE

                                     ,is_label_verify               = HP_LABEL_VERIFY
                                      #(선택) if is_label_verify is True
                                     ,label_verify_try_ceiling      = HP_LABEL_VERIFY_TRY_CEILING
                                     ,label_verify_class_min        = HP_LABEL_VERIFY_CLASS_MIN
                                     ,label_verify_ratio_max        = HP_LABEL_VERIFY_RATIO_MAX

                                      #--- options for LR image
                                     ,is_return_image_lr            = tmp_is_return_image_lr
                                      # below options can be skipped when above option is False
                                     ,scalefactor                   = HP_DG_SCALE_FACTOR
                                     ,in_path_dlc                   = PATH_BASE_IN_SUB
                                     ,in_name_dlc_csv               = HP_DG_CSV_NAME

                                      #--- increase dataset length
                                     ,in_dataset_loop               = 1

                                      #--- options for generate margin and patch
                                     ,is_force_fix                  = is_force_fix
                                     ,force_fix_size_hr             = force_fix_size_hr

                                      #--- optionas for generate tensor
                                     ,transform_img                 = transform_to_ts_img
                                     )

    if HP_VALID_WITH_PATCH:
        # valid with center-cropped patch
        update_dict_v2("", ""
                      ,"", "Valid: center-cropped patch image"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )

        dataset_val   = Custom_Dataset_V6(# Return: dict key order -> 'file_name'
                                          #                         , 'pil_img_hr', 'ts_img_hr'
                                          #                         , 'pil_lab_hr', 'ts_lab_hr'
                                          #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                          name_memo                     = ' val '
                                         ,in_path_dataset               = PATH_BASE_IN
                                         ,in_category                   = NAME_FOLDER_VAL
                                         ,in_name_folder_image          = NAME_FOLDER_IMAGES

                                          #--- options for train
                                         ,is_train                      = False

                                          #--- options for HR image
                                         ,in_path_alter_hr_image        = PATH_ALTER_HR_IMAGE

                                          #--- options for HR label
                                         ,is_return_label               = tmp_is_return_label
                                          # below options can be skipped when above option is False
                                         ,in_name_folder_label          = NAME_FOLDER_LABELS
                                         ,label_number_total            = HP_LABEL_TOTAL
                                         ,label_number_void             = HP_LABEL_VOID
                                         ,is_label_dilated              = False

                                         ,is_label_onehot_encode        = HP_LABEL_ONEHOT_ENCODE

                                         ,is_label_verify               = False

                                          #--- options for LR image
                                         ,is_return_image_lr            = tmp_is_return_image_lr
                                          # below options can be skipped when above option is False
                                         ,scalefactor                   = HP_DG_SCALE_FACTOR
                                         ,in_path_dlc                   = PATH_BASE_IN_SUB
                                         ,in_name_dlc_csv               = HP_DG_CSV_NAME

                                          #--- options for generate margin and patch
                                         ,is_force_fix                  = is_force_fix
                                         ,force_fix_size_hr             = force_fix_size_hr

                                          #--- optionas for generate tensor
                                         ,transform_img                 = transform_to_ts_img
                                         )
    else:
        # whole image valid
        update_dict_v2("", ""
                      ,"", "Valid: whole image"
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )

        dataset_val   = Custom_Dataset_V6(# Return: dict key order -> 'file_name'
                                          #                         , 'pil_img_hr', 'ts_img_hr'
                                          #                         , 'pil_lab_hr', 'ts_lab_hr'
                                          #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                          name_memo                     = ' val '
                                         ,in_path_dataset               = PATH_BASE_IN
                                         ,in_category                   = NAME_FOLDER_VAL
                                         ,in_name_folder_image          = NAME_FOLDER_IMAGES

                                          #--- options for train
                                         ,is_train                      = False

                                          #--- options for HR image
                                         ,in_path_alter_hr_image        = PATH_ALTER_HR_IMAGE

                                          #--- options for HR label
                                         ,is_return_label               = tmp_is_return_label
                                          # below options can be skipped when above option is False
                                         ,in_name_folder_label          = NAME_FOLDER_LABELS
                                         ,label_number_total            = HP_LABEL_TOTAL
                                         ,label_number_void             = HP_LABEL_VOID
                                         ,is_label_dilated              = False

                                         ,is_label_onehot_encode        = HP_LABEL_ONEHOT_ENCODE

                                         ,is_label_verify               = False

                                          #--- options for LR image
                                         ,is_return_image_lr            = tmp_is_return_image_lr
                                          # below options can be skipped when above option is False
                                         ,scalefactor                   = HP_DG_SCALE_FACTOR
                                         ,in_path_dlc                   = PATH_BASE_IN_SUB
                                         ,in_name_dlc_csv               = HP_DG_CSV_NAME

                                          #--- options for generate margin and patch
                                         ,is_force_fix                  = False

                                          #--- optionas for generate tensor
                                         ,transform_img                 = transform_to_ts_img
                                         )

    dataset_test  = Custom_Dataset_V6(# Return: dict key order -> 'file_name'
                                      #                         , 'pil_img_hr', 'ts_img_hr'
                                      #                         , 'pil_lab_hr', 'ts_lab_hr'
                                      #                         , 'pil_img_lr', 'ts_img_lr', 'info_deg'
                                      name_memo                     = 'test '
                                     ,in_path_dataset               = PATH_BASE_IN
                                     ,in_category                   = NAME_FOLDER_TEST
                                     ,in_name_folder_image          = NAME_FOLDER_IMAGES

                                      #--- options for train
                                     ,is_train                      = False

                                      #--- options for HR image
                                     ,in_path_alter_hr_image        = PATH_ALTER_HR_IMAGE

                                      #--- options for HR label
                                     ,is_return_label               = tmp_is_return_label
                                      # below options can be skipped when above option is False
                                     ,in_name_folder_label          = NAME_FOLDER_LABELS
                                     ,label_number_total            = HP_LABEL_TOTAL
                                     ,label_number_void             = HP_LABEL_VOID
                                     ,is_label_dilated              = False

                                     ,is_label_onehot_encode        = HP_LABEL_ONEHOT_ENCODE

                                     ,is_label_verify               = False

                                      #--- options for LR image
                                     ,is_return_image_lr            = tmp_is_return_image_lr
                                      # below options can be skipped when above option is False
                                     ,scalefactor                   = HP_DG_SCALE_FACTOR
                                     ,in_path_dlc                   = PATH_BASE_IN_SUB
                                     ,in_name_dlc_csv               = HP_DG_CSV_NAME

                                      #--- options for generate margin and patch
                                     ,is_force_fix                  = False

                                      #--- optionas for generate tensor
                                     ,transform_img                 = transform_to_ts_img
                                     )


    dataloader_train = DataLoader_multi_worker_FIX(dataset     = dataset_train
                                                  ,batch_size  = HP_BATCH_TRAIN
                                                  ,shuffle     = True
                                                  ,num_workers = HP_NUM_WORKERS
                                                  ,prefetch_factor = 2
                                                  ,drop_last = True
                                                  )

    if HP_VALID_WITH_PATCH:
        # center-cropped patch valid

        dataloader_val   = torch.utils.data.DataLoader(dataset     = dataset_val
                                                      ,batch_size  = HP_BATCH_VAL
                                                      ,shuffle     = False
                                                      ,num_workers = 0
                                                      )
    else:
        warnings.warn(_str)
        dataloader_val   = DataLoader_multi_worker_FIX(dataset     = dataset_val
                                                      ,batch_size  = HP_BATCH_VAL
                                                      ,shuffle     = False
                                                      ,num_workers = HP_NUM_WORKERS
                                                      ,prefetch_factor = 2
                                                      )

    dataloader_test  = DataLoader_multi_worker_FIX(dataset     = dataset_test
                                                  ,batch_size  = HP_BATCH_TEST
                                                  ,shuffle     = False
                                                  ,num_workers = HP_NUM_WORKERS
                                                  ,prefetch_factor = 2
                                                  )

    # [Train & Val & Test]-----------------------------

    #<< pkl_maker
    try:
        make_pkl = kargs['make_pkl']
    except:
        make_pkl = False

    if make_pkl:
        from _pkl_mkr.pkl_maker_0_6 import pkl_mkr_

        pkl_mkr_(dict_log_init      = dict_log_init
                ,PATH_OUT_IMAGE     = PATH_OUT_IMAGE
                ,PATH_OUT_LOG       = PATH_OUT_LOG
                ,HP_SEED            = HP_SEED
                ,HP_COLOR_MAP       = HP_COLOR_MAP
                ,dataloader_train   = dataloader_train
                ,dataloader_val     = dataloader_val
                ,dataloader_test    = dataloader_test
                )

        print("pkl completed")
        sys.exit(0)
    else:
        print("no use pkl_maker")
    #>> pkl_maker


    print("\nPause before init trainer")
    time.sleep(3)

    list_mode = ["train", "val", "test"]


    def _generate_dict_dict_log_total(list_mode, HP_DATASET_CLASSES, mode=None):
        dict_return = {list_mode[0] : {}
                      ,list_mode[1] : {}
                      ,list_mode[2] : {}
                      }

        for i_key in list_mode:
            if mode == "sr_n_ss":
                _str  = "loss_t_(" + i_key + "),loss_(" + i_key + "),PSNR_(" + i_key + "),SSIM_(" + i_key + "),"
            elif mode == "kd_sr_2_ss":
                _str  = "loss_sr_(" + i_key + "),loss_ss_(" + i_key + "),PSNR_(" + i_key + "),SSIM_(" + i_key + "),"
            elif mode == "gt_kd_sr_2_ss":
                _str = "loss_t_(" + i_key + "),loss_s_(" + i_key + "),loss_m_(" + i_key + "),PSNR_t_(" + i_key + "),SSIM_t_(" + i_key + "),PSNR_s_(" + i_key + "),SSIM_s_(" + i_key + "),"
            else:
                _str  = "loss_(" + i_key + "),PSNR_(" + i_key + "),SSIM_(" + i_key + "),"
            _str += "Pixel_Acc_(" + i_key + "),Class_Acc_(" + i_key + "),mIoU_(" + i_key + "),"
            _str += HP_DATASET_CLASSES

            update_dict_v2("epoch", _str
                          ,in_dict_dict = dict_return
                          ,in_dict_key = i_key
                          ,in_print_head = "dict_dict_log_total_" + i_key
                          ,is_print = False
                          )

        return dict_return

    dict_dict_log_total_1 = _generate_dict_dict_log_total(list_mode, HP_DATASET_CLASSES, mode="gt_kd_sr_2_ss")  

    def _generate_record_box_s(in_name, in_list):
        dict_return = {}
        for i_key in in_list:
            _str = in_name + "_" + i_key + "_"
            dict_return[i_key] = {"lr"   : RecordBox(name = _str + "lr",        is_print = False)
                                 ,"loss" : RecordBox(name = _str + "loss",      is_print = False)
                                 ,"loss_base" : RecordBox(name = _str + "loss_base",      is_print = False)
                                 ,"loss_kd" : RecordBox(name = _str + "loss_kd",      is_print = False)
                                 ,"loss_feat": RecordBox(name=_str + "loss_feat", is_print=False)

                                 }
        return dict_return

    def _generate_record_box(in_name, in_list):
        dict_return = {}
        for i_key in in_list:
            _str = in_name + "_" + i_key + "_"
            dict_return[i_key] = {"lr"   : RecordBox(name = _str + "lr",        is_print = False)
                                 ,"loss" : RecordBox(name = _str + "loss",      is_print = False)
                                 ,"psnr" : RecordBox(name = _str + "psnr",      is_print = False)
                                 ,"ssim" : RecordBox(name = _str + "ssim",      is_print = False)
                                 ,"pa"   : RecordBox(name = _str + "pixel_acc", is_print = False)
                                 ,"ca"   : RecordBox(name = _str + "class_acc", is_print = False)
                                 ,"ious" : RecordBox4IoUs(name = _str + "ious", is_print = False)
                                 }
        return dict_return

    dict_rb_t = _generate_record_box("gt_kd_sr_2_ss_t", list_mode)
    dict_rb_s = _generate_record_box("gt_kd_sr_2_ss_s", list_mode)
    dict_rb_m = _generate_record_box("gt_kd_sr_2_ss_m", list_mode)


    def ignite_eval_step(engine, batch):
        return batch
    ignite_evaluator = ignite.engine.Engine(ignite_eval_step)         
    ignite_psnr = ignite.metrics.PSNR(data_range=1.0, device=device)
    ignite_psnr.attach(ignite_evaluator, 'psnr')
    ignite_ssim = ignite.metrics.SSIM(data_range=1.0, device=device)
    ignite_ssim.attach(ignite_evaluator, 'ssim')

    dict_2_txt_v2(in_file_path = PATH_OUT_LOG
                 ,in_file_name = "log_init.csv"
                 ,in_dict = dict_log_init
                 )

    timer_trainer_start_local = time.mktime(time.localtime())
    timer_trainer_start = time.time()

    PklLoader_ = PklLoader(path_log_out = PATH_OUT_LOG
                          ,path_pkl_txt = "./_pkl_mkr/pkl_path.txt"
                          )

    # checkpoint
    checkpoint_path = kargs['checkpoint_path']
    start_epoch = 0

    if checkpoint_path:
        start_epoch = load_checkpoint(checkpoint_path, model_t, model_s, model_m, optimizer_t, optimizer_s, optimizer_m,
                                      scheduler_t, scheduler_s, scheduler_m)
        print(f"Checkpoint loaded, starting from epoch {start_epoch}")



    for i_epoch in range(start_epoch, HP_EPOCH):

        print("")
        if i_epoch > 0:                 # Time Calcurator ------------------------------------------------------------  #
            _elapsed_time = time.time() - timer_trainer_start
            try:
                _estimated_time = (_elapsed_time / i_epoch) * (HP_EPOCH)
                _tmp = time.localtime(_estimated_time + timer_trainer_start_local)
                _ETA =  str(_tmp.tm_year) + " y " + str(_tmp.tm_mon) + " m " + str(_tmp.tm_mday) + " d   "
                _ETA += str(_tmp.tm_hour) + " : " + str(_tmp.tm_min)
            except:
                _ETA = "FAIL"
        else:
            _ETA = "Calculating..."

        print("\n=== gt_kd_sr_2_ss ===\n")    
        print("Estimated Finish Time:", _ETA)

        model_t.to(device)
        model_s.to(device)
        model_m.to(device)

        is_best = None
        for i_mode in list_mode:
            print("--- init gt_kd_sr_2_ss", i_mode, "---")

            if i_mode == "test" and is_best is not None:
                if SKIP_TEST_UNTIL >= i_epoch + 1:  
                    print(SKIP_TEST_UNTIL, ": Skip the test until this epoch")
                    continue
                elif is_best == False:
                    print("Skip this epoch test")
                    continue


            if i_mode == "train":
                dataloader_input = PklLoader_.open_pkl(mode = i_mode, epoch = i_epoch + 1
                                                      ,dataloader = dataloader_train
                                                      )
            elif i_mode == "val":
                dataloader_input = PklLoader_.open_pkl(mode = i_mode, epoch = 1
                                                      ,dataloader = dataloader_val
                                                      )
            elif i_mode == "test":
                dataloader_input = PklLoader_.open_pkl(mode = i_mode, epoch = 1
                                                      ,dataloader = dataloader_test
                                                      )

            is_best =one_epoch_gt_kd_sr_2_ss(WILL_SAVE_IMAGE = WILL_SAVE_IMAGE

                                            ,CALC_WITH_LOGIT = CALC_WITH_LOGIT
                                            ,list_mode = list_mode
                                            ,i_mode    = i_mode
                                            ,HP_EPOCH  = HP_EPOCH
                                            ,i_epoch   = i_epoch

                                            ,MAX_SAVE_IMAGES  = MAX_SAVE_IMAGES
                                            ,MUST_SAVE_IMAGE  = MUST_SAVE_IMAGE
                                            ,BUFFER_SIZE      = BUFFER_SIZE
                                            ,employ_threshold = employ_threshold

                                            ,HP_LABEL_TOTAL         = HP_LABEL_TOTAL
                                            ,HP_LABEL_VOID          = HP_LABEL_VOID
                                            ,HP_DATASET_CLASSES     = HP_DATASET_CLASSES
                                            ,HP_LABEL_ONEHOT_ENCODE = HP_LABEL_ONEHOT_ENCODE
                                            ,HP_COLOR_MAP           = HP_COLOR_MAP

                                            ,PATH_OUT_MODEL = PATH_OUT_MODEL
                                            ,PATH_OUT_IMAGE = PATH_OUT_IMAGE
                                            ,PATH_OUT_LOG   = PATH_OUT_LOG

                                            ,device = device

                                            ,model_t        = model_t
                                            ,optimizer_t    = optimizer_t
                                            ,criterion_t    = criterion_t
                                            ,scheduler_t    = scheduler_t

                                            ,model_s        = model_s
                                            ,optimizer_s    = optimizer_s
                                            ,criterion_s    = criterion_s
                                            ,scheduler_s    = scheduler_s

                                            ,model_m        = model_m
                                            ,optimizer_m    = optimizer_m
                                            ,criterion_m    = criterion_m
                                            ,scheduler_m    = scheduler_m

                                            ,HP_MIXUP_A = HP_MIXUP_A

                                            ,amp_scaler = amp_scaler
                                            ,ignite_evaluator = ignite_evaluator

                                            ,dataloader_input = dataloader_input

                                            ,dict_rb_t = dict_rb_t
                                            ,dict_rb_s = dict_rb_s
                                            ,dict_rb_m = dict_rb_m

                                            ,dict_dict_log_total = dict_dict_log_total_1
                                            )


#=== End of trainer_




print("End of trainer_total.py")
