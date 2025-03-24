# main.py

if __name__ == '__main__':
    import matplotlib
    matplotlib.use("Agg")

    from model  import proposed_model, proposed_loss, proposed_loss_ss
    from utils.schedulers                       import PolyLR, Poly_Warm_Cos_LR
    from trainer   import *
    from option                     import *


    # Prevent overwriting results
    if args_options.overwrite:
        print("\n---[ Overwrite true ]---\n")
    elif os.path.isdir(PATH_BASE_OUT):
        print("[Prevent overwriting]")
        sys.exit("Prevent overwriting results function activated")

    #--- args_options.ssn

    list_opt_ss = ["a", "b", "c"]  # D3P, DABNet, CGNet

    if args_options.ssn is not None:
        if args_options.ssn not in ["D3P", "DABNet", "CGNet"]:
            _str = "Wrong parser.ssn, received " + str(args_options.ssn)
            warnings.warn(_str)
            sys.exit(-9)
        else:
            if args_options.ssn == "D3P":
                _opt_ss = list_opt_ss[0]  # D3P    -> "a"
            elif args_options.ssn == "DABNet":
                _opt_ss = list_opt_ss[1]  # DABNet -> "b"
            elif args_options.ssn == "CGNet":
                _opt_ss = list_opt_ss[2]  # CGNet  -> "c"

    else:
        _opt_ss = list_opt_ss[2]        # CGNet


    #--- args_options.kd_mode

    LIST_KD_MODE = ["kd_origin", "kd_bakd"]

    if args_options.kd_mode is not None:
        if args_options.kd_mode not in LIST_KD_MODE:
            _str = "Wrong parser.kd_mode, received " + str(args_options.kd_mode)
            warnings.warn(_str)
            sys.exit(-9)
        else:
            HP_KD_MODE = args_options.kd_mode
    else:
        HP_KD_MODE = LIST_KD_MODE[1]  # kd_bakd


    #--- args_options.mixup_a
    HP_MIXUP_A = args_options.mixup_a
    if not isinstance(HP_MIXUP_A, float):
        _str  = "parser HP_MIXUP_A must be float"
        warnings.warn(_str)
        sys.exit(-9)

    #--- args_options.will_save_image
    WILL_SAVE_IMAGE = args_options.will_save_image

    #--- args_options.skip_test_until
    SKIP_TEST_UNTIL = args_options.skip_test_until

    # --- args_options.calc_with_logit
    CALC_WITH_LOGIT = False

    #--- args_options.make_pkl
    make_pkl = args_options.make_pkl

    #--- args_options.checkpoint_path
    checkpoint_path = args_options.checkpoint_path



    if _opt_ss == list_opt_ss[0]:   # a
        from DLCs.semantic_segmentation.model_deeplab_v3_plus   import DeepLab_v3_plus
        print("(D3P) _opt_ss :", _opt_ss)
    elif _opt_ss == list_opt_ss[1]: # b
        from DLCs.semantic_segmentation.model_dabnet            import DABNet
        print("(DABNet) _opt_ss :", _opt_ss)
    elif _opt_ss == list_opt_ss[2]: # c
        from DLCs.semantic_segmentation.model_cgnet             import Context_Guided_Network as CGNet
        print("(CGNet) _opt_ss :", _opt_ss)


    #[init]----------------------------

    #log dicts reset
    dict_log_init = {}
    # set computation device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    update_dict_v2("", "---< init >---"
                  ,"", "Date: " + HP_INIT_DATE_TIME
                  ,"", "--- Dataset Parameter info ---"
                  ,"", "Dataset name: " + HP_DATASET_NAME
                  ,"", "Dataset from... " + PATH_BASE_IN
                  ,"", ""
                  ,"", "--- Hyper Parameter info ---"
                  ,"", "Device:" + str(device)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )

    # [Input HR Image]------------------------
    path_alter_hr_image = None                    

    update_dict_v2("", ""
                  ,"", "HR Image information"
                  ,"", "Original HR image"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )

    if HP_DATASET_NAME == "CamVid" or HP_DATASET_NAME == "CamVid_5Fold":
        force_fix_size_hr       = (360, 360)
        HP_BATCH_TRAIN          = 8
        HP_BATCH_VAL            = 1
        HP_BATCH_TEST           = 1
        HP_CHANNEL_HYPO         = HP_LABEL_TOTAL - 1    # 12 - 1
        HP_LABEL_ONEHOT_ENCODE  = False
        hp_augm_random_scaler   = [1.0, 1.0, 1.0, 1.25, 1.25]

    elif HP_DATASET_NAME == "MiniCity":
        force_fix_size_hr       = (660, 660)
        HP_BATCH_TRAIN          = 4
        HP_BATCH_VAL            = 1
        HP_BATCH_TEST           = 1
        HP_CHANNEL_HYPO         = HP_LABEL_TOTAL - 1    # 20 - 1
        HP_LABEL_ONEHOT_ENCODE  = False
        hp_augm_random_scaler   = [0.5, 0.75, 1.0, 1.0]


    # overwrite with argparse
    if args_options.patch_length is not None: # patch length for train
        force_fix_size_hr = (int(args_options.patch_length), int(args_options.patch_length))

    # batch_size -> batch_train / batch_val / batch_test
    if args_options.batch_train is not None:
        HP_BATCH_TRAIN = int(args_options.batch_train)
    if args_options.batch_val is not None:
        HP_BATCH_VAL   = int(args_options.batch_val)
    if args_options.batch_test is not None:
        HP_BATCH_TEST  = int(args_options.batch_test)

    HP_VALID_WITH_PATCH = args_options.valid_with_patch

    #[model_sr & Loss]------------------------

    nbb     = int(args_options.nbb // 3) # nbb = 3
    use_m1  = bool(args_options.use_m1)
    use_m2  = bool(args_options.use_m2)

    model_t = proposed_model(basic_blocks=nbb*6, use_m1=use_m1, use_m2=use_m2) # teacher (basic_blocks=18)
    model_s = proposed_model(basic_blocks=nbb*3,   use_m1=use_m1, use_m2=use_m2) # student (basic_blocks=9)

    if _opt_ss   == list_opt_ss[0]: # a
        model_m, _str = DeepLab_v3_plus(num_classes = HP_CHANNEL_HYPO, pretrained = False), "D3P (a)"
    elif _opt_ss == list_opt_ss[1]: # b
        model_m, _str = DABNet(classes=HP_CHANNEL_HYPO), "DABNet (b)"
    elif _opt_ss == list_opt_ss[2]: # c
        model_m, _str = CGNet(classes=HP_CHANNEL_HYPO, M=3, N=21), "CGNet (c)"

    update_dict_v2("", ""
                  ,"", "model info"
                  ,"", "proposed Teacher with " + str(nbb * 6) + " basic_blocks"
                  ,"", "proposed Student with " + str(nbb * 3) + " basic_blocks"
                  ,"", "use_m1 = " + str(use_m1)
                  ,"", "use_m2 = " + str(use_m2)
                  ,"", "Segmenation: " + _str
                  ,"", "pretrained = False"
                  ,"", "scale = 4"
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )

    criterion_t = proposed_loss(kd_mode=None)           # teacher

    if model_s is not None:
        criterion_s = proposed_loss(kd_mode=HP_KD_MODE)   # student
    else:
        criterion_s = None

    criterion_m = proposed_loss_ss(pred_classes    = HP_CHANNEL_HYPO
                                  ,ignore_index    = HP_LABEL_VOID
                                  )

    if criterion_s is not None:
        update_dict_v2("", "loss info"
                      ,"", "loss: proposed Loss"
                      ,"", "KD method: " + str(HP_KD_MODE)
                      ,"", "(kd_sr_2_ss) HP_MIXUP_A: " + str(HP_MIXUP_A)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )

    else:
        # Only teacher
        update_dict_v2("", "loss info"
                      ,"", "loss: proposed Loss"
                      ,"", "Only teacher training"
                      ,"", "(kd_sr_2_ss) HP_MIXUP_A: " + str(HP_MIXUP_A)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )

    is_norm_in_transform_to_tensor  = False
    HP_TS_NORM_MEAN                 = None
    HP_TS_NORM_STD                  = None

    #[optimizer]------------------------
    HP_LR_T, HP_WD_T = 1e-3, 0  # teacher
    HP_LR_S, HP_WD_S = 1e-3, 0  # student

    if _opt_ss == list_opt_ss[0]:   # a, "D3P"
        HP_LR_M, HP_WD_M          = 1e-3, 1e-9

    elif _opt_ss == list_opt_ss[1]: # b, "DABNet"
        HP_LR_M, HP_WD_M          = 2e-3, 1e-9

    elif _opt_ss == list_opt_ss[2]: # c, "CGNet"
        HP_LR_M, HP_WD_M          = 2e-3, 1e-9


    # overwrite with argparse
    if args_options.lr_t is not None:   #
        HP_LR_T = args_options.lr_t

    if args_options.wd_t is not None:   #
        HP_WD_T = args_options.wd_t

    if args_options.lr_s is not None:   #
        HP_LR_S = args_options.lr_s

    if args_options.wd_s is not None:   #
        HP_WD_S = args_options.wd_s

    if args_options.lr_m is not None:   #
        HP_LR_M = args_options.lr_m

    if args_options.wd_m is not None:   #
        HP_WD_M = args_options.wd_m

    if args_options.lr_l is not None:   #
        HP_LR_L = args_options.lr_l

    if args_options.wd_l is not None:   #
        HP_WD_L = args_options.wd_l


    optimizer_t = torch.optim.Adam(model_t.parameters(), lr = HP_LR_T, weight_decay = HP_WD_T, betas= (0.9, 0.99))  # teacher (SR)
    optimizer_s = torch.optim.Adam(model_s.parameters(), lr = HP_LR_S, weight_decay = HP_WD_S, betas= (0.9, 0.99))  # student (SR)
    optimizer_m = torch.optim.Adam(
        [
            {'params':model_m.parameters(),
             'lr':HP_LR_M,
             'weight_decay':HP_WD_M},
            {'params':model_t.parameters(),
             'lr':HP_LR_T,
             'weight_decay':HP_WD_T},
            {'params':model_s.parameters(),
             'lr':HP_LR_S,
             'weight_decay':HP_WD_S},
        ]
        , betas= (0.9, 0.99)
    )

    update_dict_v2("", ""
                  ,"", "Teacher Optimizer info"
                  ,"", "optimizer: " + "torch.optim.Adam"
                  ,"", "learning_rate: " + str(HP_LR_T)
                  ,"", "weight decay: "  + str(HP_WD_T)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )

    if optimizer_s is not None:
        update_dict_v2("", ""
                      ,"", "Student Optimizer info"
                      ,"", "optimizer: " + "torch.optim.Adam"
                      ,"", "learning_rate: " + str(HP_LR_S)
                      ,"", "weight decay: "  + str(HP_WD_S)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )

    if optimizer_m is not None:
        update_dict_v2("", ""
                      ,"", "Segmentation Optimizer info"
                      ,"", "optimizer: " + "torch.optim.Adam"
                      ,"", "learning_rate: " + str(HP_LR_M)
                      ,"", "weight decay: "  + str(HP_WD_M)
                      ,in_dict = dict_log_init
                      ,in_print_head = "dict_log_init"
                      )


    #[scheduler]----------------------------------------------
    HP_EPOCH                        = 1502 * 2
    HP_SCHEDULER_TOTAL_EPOCH        = 1500 * 2
    HP_SCHEDULER_POWER              = 0.9

    scheduler_t = PolyLR(optimizer_t, max_epoch=HP_SCHEDULER_TOTAL_EPOCH, power=HP_SCHEDULER_POWER)
    update_dict_v2("", ""
                  ,"", "Teacher scheduler info"
                  ,"", "update: epoch"
                  ,"", "scheduler: Poly " + str(HP_SCHEDULER_POWER)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )

    scheduler_s = PolyLR(optimizer_s, max_epoch=HP_SCHEDULER_TOTAL_EPOCH, power=HP_SCHEDULER_POWER)
    update_dict_v2("", ""
                  ,"", "Student scheduler info"
                  ,"", "update: epoch"
                  ,"", "scheduler: Poly " + str(HP_SCHEDULER_POWER)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )


    HP_SCHEDULER_WARM_STEPS         = 200 * 2
    HP_SCHEDULER_T_MAX              = 50  * 2
    HP_SCHEDULER_ETA_MIN            = 1e-5
    HP_SCHEDULER_STYLE              = "floor_4"

    scheduler_m = Poly_Warm_Cos_LR(optimizer_m
                                  ,warm_up_steps=HP_SCHEDULER_WARM_STEPS
                                  ,T_max=HP_SCHEDULER_T_MAX, eta_min=HP_SCHEDULER_ETA_MIN, style=HP_SCHEDULER_STYLE
                                  ,power=HP_SCHEDULER_POWER, max_epoch=HP_SCHEDULER_TOTAL_EPOCH
                                  )
    update_dict_v2("", ""
                  ,"", "Segmentation scheduler inf"
                  ,"", "Update: epoch"
                  ,"", "scheduler: Poly_Warm_Cos_LR"
                  ,"", "Power: "   + str(HP_SCHEDULER_POWER)
                  ,"", "Warm up: " + str(HP_SCHEDULER_WARM_STEPS)
                  ,"", "T max: "   + str(HP_SCHEDULER_T_MAX)
                  ,"", "ETA min: " + str(HP_SCHEDULER_ETA_MIN)
                  ,"", "Style: "   + str(HP_SCHEDULER_STYLE)
                  ,in_dict = dict_log_init
                  ,in_print_head = "dict_log_init"
                  )



    #=========================================================================================

    trainer_(

        make_pkl=make_pkl
        , checkpoint_path=checkpoint_path
        , WILL_SAVE_IMAGE=WILL_SAVE_IMAGE
        , SKIP_TEST_UNTIL=SKIP_TEST_UNTIL
        , HP_DATASET_NAME=HP_DATASET_NAME
        , HP_DATASET_CLASSES=HP_DATASET_CLASSES
        , REDUCE_SAVE_IMAGES=REDUCE_SAVE_IMAGES
        , MUST_SAVE_IMAGE=MUST_SAVE_IMAGE
        , MAX_SAVE_IMAGES=MAX_SAVE_IMAGES
        , BUFFER_SIZE=60
        , dict_log_init=dict_log_init
        , HP_SEED=HP_SEED
        , HP_EPOCH=HP_EPOCH
        , HP_BATCH_TRAIN=HP_BATCH_TRAIN
        , HP_BATCH_VAL=HP_BATCH_VAL
        , HP_BATCH_TEST=HP_BATCH_TEST
        , HP_NUM_WORKERS=HP_NUM_WORKERS

        , HP_VALID_WITH_PATCH=HP_VALID_WITH_PATCH

        , PATH_BASE_IN=PATH_BASE_IN

        , NAME_FOLDER_TRAIN=NAME_FOLDER_TRAIN
        , NAME_FOLDER_VAL=NAME_FOLDER_VAL
        , NAME_FOLDER_TEST=NAME_FOLDER_TEST
        , NAME_FOLDER_IMAGES=NAME_FOLDER_IMAGES
        , NAME_FOLDER_LABELS=NAME_FOLDER_LABELS

        , PATH_ALTER_HR_IMAGE=path_alter_hr_image

        , PATH_BASE_IN_SUB=PATH_BASE_IN_SUB

        , PATH_OUT_IMAGE=PATH_OUT_IMAGE
        , PATH_OUT_MODEL=PATH_OUT_MODEL
        , PATH_OUT_LOG=PATH_OUT_LOG

        , HP_LABEL_TOTAL=HP_LABEL_TOTAL
        , HP_LABEL_VOID=HP_LABEL_VOID
        , HP_COLOR_MAP=HP_COLOR_MAP
        , is_force_fix=True
        , force_fix_size_hr=force_fix_size_hr

        , model_t=model_t
        , optimizer_t=optimizer_t
        , scheduler_t=scheduler_t
        , criterion_t=criterion_t

        , model_s=model_s
        , optimizer_s=optimizer_s
        , scheduler_s=scheduler_s
        , criterion_s=criterion_s

        , model_m=model_m
        , optimizer_m=optimizer_m
        , scheduler_m=scheduler_m
        , criterion_m=criterion_m
        , HP_MIXUP_A=HP_MIXUP_A
        , CALC_WITH_LOGIT=CALC_WITH_LOGIT


        # <<<
        # Label Dilation
        , HP_LABEL_DILATED=False

        # Label One-Hot encoding
        , HP_LABEL_ONEHOT_ENCODE=False

        # Label Verify
        , HP_LABEL_VERIFY=HP_LABEL_VERIFY
        , HP_LABEL_VERIFY_TRY_CEILING=HP_LABEL_VERIFY_TRY_CEILING
        , HP_LABEL_VERIFY_CLASS_MIN=HP_LABEL_VERIFY_CLASS_MIN
        , HP_LABEL_VERIFY_RATIO_MAX=HP_LABEL_VERIFY_RATIO_MAX

        # DataAugm
        , HP_AUGM_RANGE_CROP_INIT=HP_AUGM_RANGE_CROP_INIT
        , HP_AUGM_ROTATION_MAX=HP_AUGM_ROTATION_MAX
        , HP_AUGM_PROB_FLIP=HP_AUGM_PROB_FLIP
        , HP_AUGM_PROB_CROP=HP_AUGM_PROB_CROP
        , HP_AUGM_PROB_ROTATE=HP_AUGM_PROB_ROTATE
        , HP_CJ_BRIGHTNESS=[1, 1]  # HP_CJ_BRIGHTNESS
        , HP_CJ_CONTRAST=[1, 1]  # HP_CJ_CONTRAST
        , HP_CJ_SATURATION=[1, 1]  # HP_CJ_SATURATION
        , HP_CJ_HUE=[0, 0]  # HP_CJ_HUE
        , HP_AUGM_RANDOM_SCALER=hp_augm_random_scaler
        # >>>

        , is_norm_in_transform_to_tensor=is_norm_in_transform_to_tensor
        , HP_TS_NORM_MEAN=HP_TS_NORM_MEAN
        , HP_TS_NORM_STD=HP_TS_NORM_STD

        # Degradation
        , HP_DG_CSV_NAME=HP_DG_CSV_NAME
        , HP_DG_SCALE_FACTOR=4
    )



    _str = "End of main.py"
    warnings.warn(_str)
    print("End of main.py")
