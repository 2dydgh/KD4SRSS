#opt.py

import time
import datetime as dt
import numpy as np
import os
import sys
import warnings
import argparse
import matplotlib

try:
    matplotlib.use("Agg")
except:
    pass

_str = "matplotlib backend: " + str(matplotlib.get_backend())
warnings.warn(_str)

RUN_WHERE = 0

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def apply_int_if_not_None(default, arg_parser, arg_parser_name="arg_parser"):
        if arg_parser is None:
            return int(default)
        else:
            if not isinstance(arg_parser, int):
                # check it again
                _str = arg_parser_name + " should be int, received " + str(arg_parser)
                warnings.warn(_str)
                sys.exit(-9)
            return arg_parser

def apply_float_if_not_None(default, arg_parser, arg_parser_name="arg_parser"):
        if arg_parser is None:
            return float(default)
        else:
            if not isinstance(arg_parser, float):
                # check it again
                _str = arg_parser_name + " should be float, received " + str(arg_parser)
                warnings.warn(_str)
                sys.exit(-9)
            return arg_parser

#--- argparse (in _opt_v2.py)
parser_options = argparse.ArgumentParser(description='_options')
# overwrite: bool (True, False)
parser_options.add_argument("--overwrite",  type = str2bool,    default = False,    help = "set this True to allow result overwrite")
# name: any string
parser_options.add_argument("--name",       type = str,         default = None,     help = "input name for result saving folder")
# dataset:
parser_options.add_argument("--dataset",    type = str,         default = None,     help = "input name for dataset")
# fold: "CamVid", "MiniCity": "A_set", "B_set" / "CamVid_5Fold": "A_set", "B_set", "C_set", "D_set", "E_set"
parser_options.add_argument("--fold",       type = str,         default = None,     help = "input name for fold")


#--- argparse (in main.py)

# SBA block
parser_options.add_argument("--nbb",        type = int,         default = 9,        help = "number of basic_blocks in proposed student SR model")
# SBAA, SCAA module
parser_options.add_argument("--use_m1",     type = str2bool,    default = True,     help = "use module 1 in proposed SR model")
parser_options.add_argument("--use_m2",     type = str2bool,    default = True,     help = "use module 2 in proposed SR model")

# Semantic segmentation network
parser_options.add_argument("--ssn",        type = str,         default = None,     help = "input name for semantic segmentation work")
# Knowledge distillation
parser_options.add_argument("--kd_mode",    type = str,         default = None,     help = "input name for knowledge distillation mode")

# Mixup
parser_options.add_argument("--mixup_a",    type = float,       default = 0.9,      help = "HP_MIXUP_A (kd_sr_2_ss의 SR(T, alpha)와 SR(S, 1-alpha)의 MIXUP 값)")

# Learning rate, Weight decay
parser_options.add_argument("--lr_t", type = float, default = None, help = "Learning Rate for Teacher")
parser_options.add_argument("--wd_t", type = float, default = None, help = "Weight Decay  for Teacher")
parser_options.add_argument("--lr_s", type = float, default = None, help = "Learning Rate for Student")
parser_options.add_argument("--wd_s", type = float, default = None, help = "Weight Decay  for Student")
parser_options.add_argument("--lr_m", type = float, default = None, help = "Learning Rate for seMantic segmentation")
parser_options.add_argument("--wd_m", type = float, default = None, help = "Weight Decay  for seMantic segmentation")
parser_options.add_argument("--lr_l", type = float, default = None, help = "Learning Rate for CN4SRSS")
parser_options.add_argument("--wd_l", type = float, default = None, help = "Weight Decay  for CN4SRSS")


# patch length for train
parser_options.add_argument("--patch_length",     type = int,      default = None, help = "One side length of square patch size for train")
parser_options.add_argument("--valid_with_patch", type = str2bool, default = True, help = "conduct validation with patch")


# batch size for train
parser_options.add_argument("--batch_train",  type = int, default = None, help = "Batch size for train")
parser_options.add_argument("--batch_val",    type = int, default = None, help = "Batch size for valid")
parser_options.add_argument("--batch_test",   type = int, default = None, help = "Batch size for test")

# save image
parser_options.add_argument("--will_save_image", type = str2bool, default = True, help = "will save result summary image?")
# skip test
parser_options.add_argument("--skip_test_until", type = int,      default = 0,    help = "skip test until given epoch")

# make pkl
parser_options.add_argument("--make_pkl", type = str2bool, default = False, help = "run pkl_mkr_ instead")

# checkpoint
parser_options.add_argument("--checkpoint_path", type = str, default = None, help="Path to the checkpoint file")



print("init _options.py")


is_sample = False


time_kr = time.gmtime(time.time() + 3600*9)
HP_INIT_DATE_TIME = time.strftime("%Y Y - %m M - %d D - %H h - %M m - %S s", time_kr)


# Dataset
HP_DATASET_NAME = "CamVid"
#HP_DATASET_NAME = "MiniCity"


#   0: computer
if RUN_WHERE == 0:
    HP_NUM_WORKERS      = 2
    PATH_BASE           = "D:/LAB/"
    NAME_FOLDER_PROJECT = "name_project"

#   1: server
elif RUN_WHERE == 1:
    HP_NUM_WORKERS      = 4
    PATH_BASE           = "your path/LAB/"
    NAME_FOLDER_PROJECT = "name_project"


#--- argparse -> parser
args_options = parser_options.parse_args()

# args_options.overwrite
if args_options.overwrite not in [True, False]:
    _str = "Wrong parser.dataset, received " + str(args_options.overwrite)
    warnings.warn(_str)
    sys.exit(-9)

# args_options.name
if args_options.name is not None:
    NAME_FOLDER_PROJECT = str(args_options.name)
    
#args_options.dataset
if args_options.dataset is not None:
    if args_options.dataset in ["CamVid", "MiniCity", "CamVid_5Fold"]:
        HP_DATASET_NAME = str(args_options.dataset)
    else:
        _str = "Wrong parser.dataset, received " + str(args_options.dataset)
        warnings.warn(_str)
        sys.exit(-9)

#args_options.fold
if args_options.fold is not None:
    if args_options.fold not in ["A_set", "B_set", "C_set", "D_set", "E_set"]:
        _str = "Wrong parser.fold, received " + str(args_options.fold)
        warnings.warn(_str)
        sys.exit(-9)
    else:
        if args_options.fold not in ["A_set", "B_set"]:
            if HP_DATASET_NAME not in ["CamVid_5Fold"]:
                _str = "Wrong parser.fold, received " + str(args_options.fold)
                _str += "\nCurrnet dataset is " + HP_DATASET_NAME
                warnings.warn(_str)
                sys.exit(-9)


#---


MAX_SAVE_IMAGES = 2

REDUCE_SAVE_IMAGES = True

_str = "Dataset name: " + HP_DATASET_NAME
warnings.warn(_str)

if HP_DATASET_NAME == "CamVid":

    MUST_SAVE_IMAGE = ["0016E5_08123.png"    # CamVid 12 v4 Fold A
                      ,"0016E5_08145.png"    # CamVid 12 v4 Fold B
                      ,"0016E5_08155.png"    # CamVid 12 v4 Fold B
                      ,"0016E5_07959.png"    # CamVid 12 v4 Fold A
                      ,"0016E5_08131.png"    # CamVid 12 v4 Fold A
                      ]
    
    if args_options.fold is not None:
        NAME_FOLDER_DATASET = "CamVid_12_2Fold_v4/" + args_options.fold
    else:
        NAME_FOLDER_DATASET = "CamVid_12_2Fold_v4/A_set"

    NAME_FOLDER_ALTER_HR_IAMGE = None

    NAME_FOLDER_DATASET_SUB = "CamVid_12_DLC_v1/x4_BILINEAR"
    
    HP_LABEL_VERIFY                = False
    HP_LABEL_VERIFY_TRY_CEILING    = 10
    HP_LABEL_VERIFY_CLASS_MIN      = 6
    HP_LABEL_VERIFY_RATIO_MAX      = 0.6
    
elif HP_DATASET_NAME == "MiniCity":
    
    MUST_SAVE_IMAGE = ["aachen_000045_000019.png"       # MiniCity 19 v1 Fold A
                      ,"frankfurt_000001_056580.png"    # MiniCity 19 v1 Fold A
                      ,"ulm_000030_000019.png"          # MiniCity 19 v1 Fold B
                      ]
    
    if args_options.fold is not None:
        NAME_FOLDER_DATASET = "MiniCity_19_2Fold_v1/" + args_options.fold
    else:
        NAME_FOLDER_DATASET = "MiniCity_19_2Fold_v1/A_set"

    NAME_FOLDER_ALTER_HR_IAMGE = None
    
    NAME_FOLDER_DATASET_SUB = "MiniCity_19_DLC_v1/x4_BILINEAR"
    
    HP_LABEL_VERIFY                = True
    HP_LABEL_VERIFY_TRY_CEILING    = 10
    HP_LABEL_VERIFY_CLASS_MIN      = 6
    HP_LABEL_VERIFY_RATIO_MAX      = 0.6
    

elif HP_DATASET_NAME == "CamVid_5Fold":
    
    MUST_SAVE_IMAGE = ["0016E5_08123.png"    # CamVid 5 Fold v1 Fold A
                      ,"0016E5_08155.png"    # CamVid 5 Fold v1 Fold B
                      ,"0016E5_08147.png"    # CamVid 5 Fold v1 Fold C
                      ,"0016E5_08159.png"    # CamVid 5 Fold v1 Fold D
                      ,"0016E5_08151.png"    # CamVid 5 Fold v1 Fold E
                      ]
    
    if args_options.fold is not None:
        NAME_FOLDER_DATASET = "CamVid_12_5Fold_v1/" + args_options.fold
    else:
        NAME_FOLDER_DATASET = "CamVid_12_5Fold_v1/A_set"

    NAME_FOLDER_ALTER_HR_IAMGE = None
    
    NAME_FOLDER_DATASET_SUB = "CamVid_12_DLC_v1/x4_BILINEAR"
    
    HP_LABEL_VERIFY                = False
    HP_LABEL_VERIFY_TRY_CEILING    = 10
    HP_LABEL_VERIFY_CLASS_MIN      = 6
    HP_LABEL_VERIFY_RATIO_MAX      = 0.6
    
else:
    sys.exit("There is no such dataset")

NAME_FOLDER_PROJECT += "_" + NAME_FOLDER_DATASET.split("/")[0] + "_" + NAME_FOLDER_DATASET.split("/")[-1]

if is_sample:
    print("=== === === [Sample Run] === === ===")
    NAME_FOLDER_DATASET = "Sample_set/A_set"
    NAME_FOLDER_ALTER_HR_IAMGE = None
    NAME_FOLDER_DATASET_SUB = "Sample_set_DLC/x4_BILINEAR"
    NAME_FOLDER_PROJECT = "Sample_set_A_set"

NAME_FOLDER_TRAIN   = "train"
NAME_FOLDER_VAL     = "val"
NAME_FOLDER_TEST    = "test"

NAME_FOLDER_IMAGES  = "images"
NAME_FOLDER_LABELS  = "labels"

# make folder named _RUN_on_{number} below the LAB folder
if not os.path.isdir(PATH_BASE + "_RUN_on_" + str(RUN_WHERE)):
    print("Wrong [RUN_WHERE] input:", RUN_WHERE)
    sys.exit(9)
else:
    print("RUN on", RUN_WHERE, PATH_BASE)


PATH_BASE_IN            = PATH_BASE + "datasets/project_use/" + NAME_FOLDER_DATASET         + "/"
try:
    PATH_ALTER_HR_IMAGE = PATH_BASE + "datasets/project_use/" + NAME_FOLDER_ALTER_HR_IAMGE  + "/"
except:
    PATH_ALTER_HR_IMAGE = None
PATH_BASE_IN_SUB        = PATH_BASE + "datasets/project_use/" + NAME_FOLDER_DATASET_SUB     + "/"
PATH_BASE_OUT           = PATH_BASE + "result_files/"         + NAME_FOLDER_PROJECT         + "/"


PATH_OUT_LOG = PATH_BASE_OUT + "logs/"
PATH_OUT_MODEL = PATH_BASE_OUT + "models/"
PATH_OUT_IMAGE = PATH_BASE_OUT + "images/"


#[hyper parameters]----------------------------------------


HP_SEED = 15

if HP_DATASET_NAME == "CamVid":
    
    HP_LABEL_TOTAL          = 12
    HP_LABEL_VOID           = 11
    HP_LABEL_ONEHOT_ENCODE  = False
    
    HP_DATASET_CLASSES = ("0(Sky),1(Building),2(Column_pole),3(Road),4(Sidewalk),5(Tree),"
                         +"6(SignSymbol),7(Fence),8(Car),9(Pedestrian),10(Bicyclist)"
                         )

    HP_COLOR_MAP = {0:  [128, 128, 128]     # 00 Sky
                   ,1:  [128,   0,   0]     # 01 Building
                   ,2:  [192, 192, 128]     # 02 Column_pole
                   ,3:  [128,  64, 128]     # 03 Road
                   ,4:  [  0,   0, 192]     # 04 Sidewalk
                   ,5:  [128, 128,   0]     # 05 Tree
                   ,6:  [192, 128, 128]     # 06 SignSymbol
                   ,7:  [ 64,  64, 128]     # 07 Fence
                   ,8:  [ 64,   0, 128]     # 08 Car
                   ,9:  [ 64,  64,   0]     # 09 Pedestrian
                   ,10: [  0, 128, 192]     # 10 Bicyclist
                   ,11: [  0,   0,   0]     # 11 Void
                   }
    
elif HP_DATASET_NAME == "MiniCity":
    
    #HP_ORIGIN_IMG_W, HP_ORIGIN_IMG_H = 2048, 1024
    
    HP_LABEL_TOTAL          = 20
    HP_LABEL_VOID           = 19
    HP_LABEL_ONEHOT_ENCODE  = False
    
    HP_DATASET_CLASSES = ("0(Road),1(Sidewalk),2(Building),3(Wall),4(Fence),"
                         +"5(Pole),6(Traffic_light),7(Traffic_sign),8(Vegetation),9(Terrain),"
                         +"10(Sky),11(Person),12(Rider),13(Car),14(Truck),"
                         +"15(Bus),16(Train),17(Motorcycle),18(Bicycle)"
                         )
    
    HP_COLOR_MAP = {0:  [128,  64, 128]     # 00 Road
                   ,1:  [244,  35, 232]     # 01 Sidewalk
                   ,2:  [ 70,  70,  70]     # 02 Building
                   ,3:  [102, 102, 156]     # 03 Wall
                   ,4:  [190, 153, 153]     # 04 Fence
                   ,5:  [153, 153, 153]     # 05 Pole
                   ,6:  [250, 170,  30]     # 06 Traffic light
                   ,7:  [220, 220,   0]     # 07 Traffic sign
                   ,8:  [107, 142,  35]     # 08 Vegetation
                   ,9:  [152, 251, 152]     # 09 Terrain
                   ,10: [ 70, 130, 180]     # 10 Sky
                   ,11: [220,  20,  60]     # 11 Person
                   ,12: [255,   0,   0]     # 12 Rider
                   ,13: [  0,   0, 142]     # 13 Car
                   ,14: [  0,   0,  70]     # 14 Truck
                   ,15: [  0,  60, 100]     # 15 Bus
                   ,16: [  0,  80, 100]     # 16 Train
                   ,17: [  0,   0, 230]     # 17 Motorcycle
                   ,18: [119,  11,  32]     # 18 Bicycle
                   ,19: [  0,   0,   0]     # 19 Void
                   }

elif HP_DATASET_NAME == "CamVid_5Fold":

    HP_LABEL_TOTAL          = 12
    HP_LABEL_VOID           = 11
    HP_LABEL_ONEHOT_ENCODE  = False
    
    HP_DATASET_CLASSES = ("0(Sky),1(Building),2(Column_pole),3(Road),4(Sidewalk),5(Tree),"
                         +"6(SignSymbol),7(Fence),8(Car),9(Pedestrian),10(Bicyclist)"
                         )
    
    HP_COLOR_MAP = {0:  [128, 128, 128]     # 00 Sky
                   ,1:  [128,   0,   0]     # 01 Building
                   ,2:  [192, 192, 128]     # 02 Column_pole
                   ,3:  [128,  64, 128]     # 03 Road
                   ,4:  [  0,   0, 192]     # 04 Sidewalk
                   ,5:  [128, 128,   0]     # 05 Tree
                   ,6:  [192, 128, 128]     # 06 SignSymbol
                   ,7:  [ 64,  64, 128]     # 07 Fence
                   ,8:  [ 64,   0, 128]     # 08 Car
                   ,9:  [ 64,  64,   0]     # 09 Pedestrian
                   ,10: [  0, 128, 192]     # 10 Bicyclist
                   ,11: [  0,   0,   0]     # 11 Void
                   }


#---(HP 7) Data Augm- & ColorJitter
HP_AUGM_RANGE_CROP_INIT = (4, 10)
HP_AUGM_ROTATION_MAX = 5
HP_AUGM_PROB_FLIP = 50
HP_AUGM_PROB_CROP = 70
HP_AUGM_PROB_ROTATE = 20

#torchvision transform ColorJitter
HP_CJ_BRIGHTNESS = (0.6, 1.4)
HP_CJ_CONTRAST   = (0.7, 1.2)
HP_CJ_SATURATION = (0.9, 1.3)
HP_CJ_HUE        = (-0.05, 0.05)

#---(HP 9) Degradation
if HP_DATASET_NAME == "CamVid":
    HP_DG_CSV_NAME = "degradation_2.csv"
    
elif HP_DATASET_NAME == "MiniCity":
    HP_DG_CSV_NAME = "degradation_MiniCity.csv"
    
elif HP_DATASET_NAME == "CamVid_5Fold":
    HP_DG_CSV_NAME = "degradation_2.csv" 

HP_DG_CSV_PATH = PATH_BASE_IN_SUB + HP_DG_CSV_NAME

# Scale Factor
HP_DG_SCALE_FACTOR = 4

# resize
HP_DG_RESIZE_OPTION = "BILINEAR"

# Gaussian noise sigma
HP_DG_RANGE_NOISE_SIGMA = (1, 30)

# noise
HP_DG_NOISE_GRAY_PROB = 40


print("inputs...")
print("dataset (PATH_BASE_IN):", PATH_BASE_IN)
print("degradation_2 (HP_DG_CSV_PATH): ", HP_DG_CSV_PATH)

print("outputs...")
print("logs (PATH_OUT_LOG):", PATH_OUT_LOG)
print("models (PATH_OUT_MODEL):", PATH_OUT_MODEL)
print("images (PATH_OUT_IMAGE):", PATH_OUT_IMAGE)


print("EOF: _opt_v2.py")


