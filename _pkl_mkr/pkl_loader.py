"""
pkl_loader
v 0.1
    init
    matches with pkl_maker v 0.5

v 0.2
    pkl_loader 사용기록 log 파일 생성기능 추가

v 0.3
    val 에 대해 특정 pkl 지정 기능 추가

v 0.4
    log에 time stamp 추가
"""

import os
import sys
import warnings

import time
import pickle

_version = "pkl_loader v 0.4"

warnings.warn(_version)

class LogText():
    def __init__(self, name, path_out):
        self.list_str = []
        if path_out[-1] == "/":
            self.path_out = path_out[:-1]
        else:
            self.path_out = path_out
        
        if self.path_out is None:
            self.path_txt = None
            print("Log for", name, "will not be saved.")
        else:
            self.path_txt = self.path_out + "/log_" + name + ".txt"
            print("Log for", name, "will be saved at:", self.path_txt)
        
    
    def add_str(self, *args):
        in_str = None
        for i_arg in args:
            if in_str is not None:
                in_str += " " + str(i_arg)
            else:
                in_str = str(i_arg)
        print(in_str)
        self.list_str.append(in_str)
    
    def save_txt(self):
        if not os.path.exists(self.path_out):
            os.makedirs(self.path_out)
        with open(self.path_txt, mode="w") as _txt:
            for i_str in self.list_str:
                _txt.write(i_str + "\n")
        print("Log saved:", self.path_txt)
        
    
    def update_txt(self, *args):
        if not os.path.exists(self.path_out):
            os.makedirs(self.path_out)
        
        if not os.path.exists(self.path_txt):
            _mode = "w"
        else:
            _mode = "a"
        
        in_str = None
        for i_arg in args:
            if in_str is not None:
                in_str += " " + str(i_arg)
            else:
                in_str = str(i_arg)
        
        time_kr = time.gmtime(time.time() + 3600*9) # 한국 표준시 KST = 협정 세계시 UTC + 09:00
        _time_stamp = time.strftime("%Y Y - %m M - %d D - %H h - %M m - %S s", time_kr)
        
        in_str += " --- " + _time_stamp
        
        print(in_str)
        
        with open(self.path_txt, mode=_mode) as _txt:
            _txt.write(in_str + "\n")
        
        # print("Log updated:", self.path_txt)

class PklLoader():
    def __init__(self, **kwargs):
        try:
            if str(kwargs['path_log_out'])[-1] == "/":
                path_log_out = str(kwargs['path_log_out'])[:-1]
            else:
                path_log_out = str(kwargs['path_log_out'])
        except:
            # do not write any log
            path_log_out = None
        
        self.LogText_ = LogText(name = "PklLoader", path_out = path_log_out)   # pkl_loader log 파일 생성용
        
        _HEAD = "[ PklLoader ]"
        default_path_pkl_txt = "./_pkl_mkr/pkl_path.txt"
        try:
            path_pkl_txt = str(kwargs['path_pkl_txt'])
        except:
            path_pkl_txt = default_path_pkl_txt
        
        self.LogText_.add_str(_HEAD, "path_pkl_txt:", path_pkl_txt)
        
        try:
            path_pkl_key = str(kwargs['path_pkl_key'])
        except:
            path_pkl_key = None
            try:
                if path_log_out is not None:
                    _str = path_log_out.split("/")[-2]
                    
                    for i_str in ["CamVid_12_2Fold_v4_A_set", "CamVid_12_2Fold_v4_B_set"
                                 ,"MiniCity_19_2Fold_v1_A_set", "MiniCity_19_2Fold_v1_B_set"
                                 ]:
                        if i_str in _str:
                            path_pkl_key = i_str
                            break
                        
            except:
                pass
        
        self.LogText_.add_str(_HEAD, "path_pkl_key:", path_pkl_key)
        
        if path_log_out is None and path_pkl_key is None:
            self.LogText_.add_str(_HEAD, "both path_log_out and path_pkl_key can not be None.")
            self.LogText_.save_txt()
            sys.exit(-9)
        
        if not os.path.exists(path_pkl_txt):
            # pickle 안씀
            self.use_pkl = False
            warnings.warn(_HEAD + " File not found: " + path_pkl_txt)
            self.LogText_.add_str(_HEAD, "will not use pickle.")
        else:
            # pickle 씀
            self.LogText_.add_str(_HEAD, "path_pkl_txt:", path_pkl_txt)
            self.use_pkl = True
            
        
        if self.use_pkl:
            # pickle 씀
            dict_path = {}
            with open(path_pkl_txt, mode="r") as _txt:
                _lines = _txt.readlines() # raw lines
                # close file
            
            split_word = "==="
            lines = [] # lines without comment
            for i_line in _lines:
                _line = i_line.strip("\n")
                is_comment = False
                try:
                    if _line[0] == "#":
                        # comment line
                        is_comment = True
                except:
                    # empty line
                    is_comment = True
                    pass
                    
                if not is_comment:
                    #print(_line)
                    lines.append(_line)
            
            # find split word
            if "key" in lines[0] and "value" in lines[0]:
                _str = lines[0]
                _str = _str.strip("key")
                _str = _str.strip("value")
                if len(_str) > 0:
                    split_word = _str
                else:
                    self.LogText_.add_str(_HEAD, "wrong split word: (", _str, "). Default option used instead.")
                    
                
            else:
                self.LogText_.add_str(_HEAD, "no split word style option found. This option should be at first line except comment.")
                self.LogText_.add_str(_HEAD, "example: key===value")
                self.LogText_.add_str(_HEAD, "current first line:", lines[0])
            
            self.LogText_.add_str(_HEAD, "split_word:", split_word)
            
            for i_line in lines:
                _list = i_line.split(split_word)
                if _list[0] != "key" and len(_list) == 2:
                    dict_path[_list[0]] = _list[1]
            
            self.LogText_.add_str(_HEAD, "identified paths:", dict_path)
            self.path_pkl = dict_path[path_pkl_key]
            self.LogText_.add_str(_HEAD, "selected path key:", path_pkl_key)
            self.LogText_.add_str(_HEAD, "selected path value:", self.path_pkl)
            
            path_val_pkl_key =  "val_" + path_pkl_key
            self.LogText_.add_str(_HEAD, "check designated path for val pkl:", path_val_pkl_key)
            try:
                self.path_val_pkl = dict_path[path_val_pkl_key]
                if self.path_val_pkl == "None":
                    self.path_val_pkl = None
            except:
                self.path_val_pkl = None
            self.LogText_.add_str(_HEAD, "detected path for val pkl:", str(self.path_val_pkl))
            
            _count = 0
            for i_file in os.listdir(self.path_pkl):
                if ".pkl" in i_file:
                    _count += 1
            
            if _count == 0:
                self.LogText_.add_str(_HEAD, "no pickles found. Check path again.")
                self.LogText_.save_txt()
                sys.exit(-9)
            else:
                self.LogText_.add_str(_HEAD, "pickles found:", _count)
            
        else:
            # pickle 안씀
            self.LogText_.add_str(_HEAD, "will use pytorch dataloader instead.")
        
        self.LogText_.add_str(_HEAD, "init finished.\n")
        self.LogText_.save_txt()
    
    def open_pkl(self, **kwargs):
        _HEAD = "[ PklLoader ]"
        _pkl_body = "_E_"
        
        # force_not_use_pkl
        try:
            _force_use_dl = kwargs['force_use_dl']
        except:
            _force_use_dl = False
        #   force use kwargs['dataloader'] (pytorch dataloader) instead.
        
        # dataloader
        # _dataloader = kwargs['dataloader']
        #   pytorch dataloader.
        #   this will be used when pkl is not used.
        
        # mode
        _mode = str(kwargs['mode'])
        #   one of ["train", "val", "test"]
        #   this will be used as pkl name head
        
        # epoch
        _epoch = str(kwargs['epoch'])
        #   current epoch
        #   this will be used as pkl name tail
        
        if not self.use_pkl or _force_use_dl:
            self.LogText_.update_txt(_HEAD, "for", _mode, _epoch, ", use pytorch dataloader instead.")
            # return _dataloader
            return kwargs['dataloader']
        else:
            if _mode == "val" and self.path_val_pkl is not None:
                #@@ 지정경로의 pkl 불러오기
                _path = self.path_val_pkl
                self.LogText_.update_txt(_HEAD, "for val, use designated pkl for dataloader:", _path)
                
            else:
                # default pkl 불러오기
                _path = self.path_pkl + "/" + _mode + _pkl_body + _epoch + ".pkl"
                self.LogText_.update_txt(_HEAD, "for", _mode, _epoch, "use pkl for dataloader:", _path)
            
            try:
                with open(_path, mode="rb") as _pkl:
                    dataloader_pkl = pickle.load(_pkl)
                return dataloader_pkl
            except:
                _str = _HEAD + " pickle not found: " + _path
                self.LogText_.update_txt(_str)
                warnings.warn(_str)
                sys.exit(-9)



if __name__ == "__main__":
    import time
    # path_out_log = "C:/LAB/result_files/_debug_CamVid_12_2Fold_v4_A_set/logs"
    path_out_log = "C:/LAB/result_files/_debug_MiniCity_19_2Fold_v1_A_set/logs"
    path_pkl_txt = "./pkl_path.txt"
    
    PklLoader_ = PklLoader(path_log_out = path_out_log
                          ,path_pkl_txt = path_pkl_txt
                          )
    _torch_dl = "blabla" # pytorch dataloader
    for i_epoch in range(10):
        for i_mode in ["train", "val", "test"]:
            _timer = time.time()
            _dl = PklLoader_.open_pkl(mode = i_mode, epoch = i_epoch + 1
                                    ,dataloader = _torch_dl
                                    ,force_use_dl = i_mode=="test"
                                    )
            
            print("load took", time.time() - _timer, "sec")
            _count = 0
            _count_max = len(_dl)
            for i_items in _dl:
                _count += 1
                print("\r", i_mode, _count, "/", _count_max, end="")
                
            print("\nlast file_names in i_items", i_items[0])
    
    print("EOF: pkl_loader")

