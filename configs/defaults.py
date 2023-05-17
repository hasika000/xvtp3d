import os

from yacs.config import CfgNode as CN

_C = CN()

# ------ GENERAL ---
_C.save_dir = 'save_model'
_C.save_tag = 'run'
_C.device = 'cuda:0'
_C.val_include = False
_C.test_include = True
_C.num_epochs = 1
_C.save_model = True
_C.save_interval = 100
_C.modality = 'bev'
_C.print_interval = 1
_C.distributed = 1
_C.fpv_type = 'ego'

# ----- DATA ---
_C.DATA = CN()
_C.DATA.obs_len = 1
_C.DATA.pred_len = 1
_C.DATA.skip = 1
_C.DATA.batch_size = 1
_C.DATA.augment = True
_C.DATA.subset = True
_C.DATA.split = False

# ----- DATA_LOADER ---
_C.DATALOADER = CN()
_C.DATALOADER.data_dir = 'dir'
_C.DATALOADER.calib_path = 'dir'
_C.DATALOADER.num_worker = 0
_C.DATALOADER.shuffle = True
_C.DATALOADER.delim = 'tab'

# ----- MODEL ---
_C.MODEL = CN()
_C.MODEL.hidden_size = 128
_C.MODEL.mode_num = 1
_C.MODEL.gt_decoder = False
_C.MODEL.cross_type = 1
_C.MODEL.random_mask = 0.0
_C.MODEL.cross_enc = False
_C.MODEL.self_rein = False
_C.MODEL.share_weight = True

# ----- SOLVER ---
_C.SOLVER = CN()
_C.SOLVER.lr = 0.001
_C.SOLVER.bev_weight = 1.0
_C.SOLVER.fpv_weight = 1.0
_C.SOLVER.step_size = 10
_C.SOLVER.scheduler_lr_type = 'p'
_C.SOLVER.grad_clip = 0.01
_C.SOLVER.patience = 10
_C.SOLVER.milestones = (10,)

# ----- EVAL ---
_C.EVAL = CN()
_C.EVAL.optimization = True
_C.EVAL.MRminFDE = -1
_C.EVAL.cnt_sample = -1
_C.EVAL.opti_time = 0.1
_C.EVAL.core_num = 1


