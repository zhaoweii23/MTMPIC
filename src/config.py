import os
import subprocess
from typing import Dict
import logging
import torch
import random
import numpy as np

class GlobalConfig:
    DATA_PATH = "data/raw/IONCHADATA.csv"
    REQUIRED_COLUMNS = ['SMILES', 'Sequence', 'ABB', 'Label']
    DATA_CACHE_DIR = "data/processed/cache/"
    PDB_file_path = "data/pdb/"
    PDB_cache_path = "pdb_cache/"
    CACHE_DIR = "feature_cache/"
    CACHE_DIR_test = "feature_cache_test/"
    CACHE_DIR_test_val = "feature_cache_ESM2_val_test/"
    CACHE_DIR_train = "feature_cache_ESM2_train_New/" 
    CACHE_DIR_test= "feature_cache_ESM2_test_New/"
    CACHE_DIR_val = "feature_cache_ESM2_val_New/"
    
    FP_SIZE = 2048
    
    # 多分类标签设置 (0: 无活性, 1: 抑制, 2: 激活)
    LABEL_MAPPING = {
        0: "inactive",
        1: "inhibitor",
        2: "activator"
    }
    NUM_CLASSES = len(LABEL_MAPPING)
    
    # 模型参数
    MOL_FEAT_DIM = 2048  # Morgan指纹维度
    PROT_FEAT_DIM = 4096  # ProtBERT特征维度
    HIDDEN_DIM = 2048
    DROPOUT_RATE = 0.3
    
    # 训练参数
    SEED = 42
    TEST_SIZE = 0.2
    BATCH_SIZES = {
        'high_volume': 256,
        'mid_volume': 128,
        'low_volume': 64,
        'tiny_volume': 32
    }
    LEARNING_RATES = {
        'base': 1e-4,
        'high': 3e-4,
        'mid': 1e-4,
        'low': 5e-5,
        'tiny': 1e-5
    }
    WEIGHT_DECAY = 1e-5
    EPOCHS = 150
    EARLY_STOP_PATIENCE = 15
    GRAD_CLIP = 1.0
    
    # 设备配置
    @classmethod
    def get_free_gpus(cls, utilization_threshold=10):
        try:
            output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )
            gpu_utils = [int(x) for x in output.strip().split('\n')]
            free_gpus = [i for i, util in enumerate(gpu_utils) if util < utilization_threshold]
            
            # 记录每个GPU的利用率
            for i, util in enumerate(gpu_utils):
                logging.info(f"GPU {i} 当前利用率: {util}%")
            
            return free_gpus
        except (subprocess.CalledProcessError, FileNotFoundError):
            logging.warning("无法获取GPU信息，可能是nvidia-smi不可用")
            return []

    @classmethod
    def setup_gpu_environment(cls):
        """设置GPU环境，只使用空闲的GPU"""
        if not torch.cuda.is_available():
            cls.DEVICE = torch.device("cuda")
            cls.NUM_GPUS = 0
            return

        # 获取空闲GPU（利用率低的GPU）
        free_gpus = cls.get_free_gpus(utilization_threshold=1)
        if free_gpus:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(free_gpus[0])
            cls.DEVICE = torch.device(f"cuda:{free_gpus[0]}")
            cls.NUM_GPUS = 1
            logging.info(f"utilization < 1% 的GPU: {free_gpus[0]}")
        else:
            logging.warning("没有空闲的GPU可用，使用CPU进行训练")
            cls.DEVICE = torch.device("cpu")
            cls.NUM_GPUS = 0
            return

    DEVICE = torch.device("cuda")  # 初始值
    NUM_GPUS = 0  # 初始值，将在setup_gpu_environment中更新
    NUM_WORKERS = min(4, os.cpu_count())
    
    # 日志和保存
    SAVE_DIR = "saved_models/"
    LOG_DIR = "logs/"
    PLOT_DIR = "plots/"
    
    @classmethod
    def setup_dirs(cls):
        """创建必要的目录"""
        os.makedirs(cls.DATA_CACHE_DIR, exist_ok=True)
        os.makedirs(cls.SAVE_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.PLOT_DIR, exist_ok=True)
    
    @classmethod
    def set_seed(cls, seed=None):
        """设置随机种子"""
        seed = seed or cls.SEED
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
