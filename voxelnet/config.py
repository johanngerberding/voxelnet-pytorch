from yacs.config import CfgNode as CN


_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_WORKERS = 4

_C.DATA = CN()
_C.DATA.DIR= "/data/kitti/3d_vision/data/MD_KITTI"
_C.DATA.CALIB_DIR= "/data/kitti/3d_vision/data/KITTI/training/calib"


_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.NUM_WORKERS = 8
_C.TRAIN.LR = 0.01 
_C.TRAIN.LR_SCHEDULER_STEP = 150
_C.TRAIN.ALPHA = 1.5
_C.TRAIN.BETA = 1
_C.TRAIN.NUM_EPOCHS = 10  
_C.TRAIN.GRADIENT_CLIP = 5

_C.VAL = CN()
_C.VAL.BATCH_SIZE = 1
_C.VAL.NUM_WORKERS = 4


_C.IMAGE = CN()
_C.IMAGE.WIDTH = 1242 
_C.IMAGE.HEIGHT = 375 
_C.IMAGE.CHANNELS = 3 

_C.OBJECT = CN()
_C.OBJECT.NAME = "Car"
if _C.OBJECT.NAME == "Car":
    _C.OBJECT.Z_MIN = -3
    _C.OBJECT.Z_MAX = 1
    _C.OBJECT.Y_MIN = -40 
    _C.OBJECT.Y_MAX = 40 
    _C.OBJECT.X_MIN = 0 
    _C.OBJECT.X_MAX = 70.4 
    _C.OBJECT.Z_VOXEL_SIZE = 0.4 
    _C.OBJECT.Y_VOXEL_SIZE = 0.2 
    _C.OBJECT.X_VOXEL_SIZE = 0.2 
    _C.OBJECT.POINTS_PER_VOXEL = 35 
    _C.OBJECT.DEPTH = int((_C.OBJECT.Z_MAX - _C.OBJECT.Z_MIN) / _C.OBJECT.Z_VOXEL_SIZE) 
    _C.OBJECT.HEIGHT = int((_C.OBJECT.Y_MAX - _C.OBJECT.Y_MIN) / _C.OBJECT.Y_VOXEL_SIZE) 
    _C.OBJECT.WIDTH = int((_C.OBJECT.X_MAX - _C.OBJECT.X_MIN) / _C.OBJECT.X_VOXEL_SIZE)
    _C.OBJECT.FEATURE_RATIO = 2
    _C.OBJECT.FEATURE_WIDTH = int(_C.OBJECT.WIDTH / _C.OBJECT.FEATURE_RATIO)
    _C.OBJECT.FEATURE_HEIGHT = int(_C.OBJECT.HEIGHT / _C.OBJECT.FEATURE_RATIO)
    _C.OBJECT.ANCHOR_L = 3.9
    _C.OBJECT.ANCHOR_W = 1.6
    _C.OBJECT.ANCHOR_H = 1.56
    _C.OBJECT.ANCHOR_Z = -1.0 - _C.OBJECT.ANCHOR_H / 2
    _C.OBJECT.RPN_POS_IOU = 0.6
    _C.OBJECT.RPN_NEG_IOU = 0.45 
else:
    _C.OBJECT.Z_MIN = -3
    _C.OBJECT.Z_MAX = 1
    _C.OBJECT.Y_MIN = -20 
    _C.OBJECT.Y_MAX = 20 
    _C.OBJECT.X_MIN = 0 
    _C.OBJECT.X_MAX = 48 
    _C.OBJECT.Z_VOXEL_SIZE = 0.4 
    _C.OBJECT.Y_VOXEL_SIZE = 0.2 
    _C.OBJECT.X_VOXEL_SIZE = 0.2 
    _C.OBJECT.POINTS_PER_VOXEL = 45 
    _C.OBJECT.DEPTH = int((_C.OBJECT.Z_MAX - _C.OBJECT.Z_MIN) / _C.OBJECT.Z_VOXEL_SIZE) 
    _C.OBJECT.HEIGHT = int((_C.OBJECT.Y_MAX - _C.OBJECT.Y_MIN) / _C.OBJECT.Y_VOXEL_SIZE) 
    _C.OBJECT.WIDTH = int((_C.OBJECT.X_MAX - _C.OBJECT.X_MIN) / _C.OBJECT.X_VOXEL_SIZE)
    _C.OBJECT.FEATURE_RATIO = 2
    _C.OBJECT.FEATURE_WIDTH = int(_C.OBJECT.WIDTH / _C.OBJECT.FEATURE_RATIO)
    _C.OBJECT.FEATURE_HEIGHT = int(_C.OBJECT.HEIGHT / _C.OBJECT.FEATURE_RATIO)
    
if _C.OBJECT.NAME == 'Pedestrian':    
    _C.OBJECT.ANCHOR_L = 0.8 
    _C.OBJECT.ANCHOR_W = 0.6
    _C.OBJECT.ANCHOR_H = 1.73
    _C.OBJECT.ANCHOR_Z = -0.6 - _C.OBJECT.ANCHOR_H / 2
    _C.OBJECT.RPN_POS_IOU = 0.5
    _C.OBJECT.RPN_NEG_IOU = 0.35

if _C.OBJECT.NAME == 'Cyclist':    
    _C.OBJECT.ANCHOR_L = 1.76 
    _C.OBJECT.ANCHOR_W = 0.6
    _C.OBJECT.ANCHOR_H = 1.73
    _C.OBJECT.ANCHOR_Z = -0.6 - _C.OBJECT.ANCHOR_H / 2
    _C.OBJECT.RPN_POS_IOU = 0.5
    _C.OBJECT.RPN_NEG_IOU = 0.35


_C.RPN = CN()
_C.RPN.NMS_POST_TOPK = 20 
_C.RPN.NMS_THRES = 0.1 
_C.RPN.SCORE_THRES = 0.96 


# Mean from kitti dataset
_C.CALIB = CN()
_C.CALIB.T_VELO_2_CAM = ([
    [7.49916597e-03, -9.99971248e-01, -8.65110297e-04, -6.71807577e-03],
    [1.18652889e-02, 9.54520517e-04, -9.99910318e-01, -7.33152811e-02],
    [9.99882833e-01, 7.49141178e-03, 1.18719929e-02, -2.78557062e-01],
    [0, 0, 0, 1]
]) 
_C.CALIB.R_RECT_0 = ([
    [0.99992475, 0.00975976, -0.00734152, 0],
    [-0.0097913, 0.99994262, -0.00430371, 0],
    [0.00729911, 0.0043753, 0.99996319, 0],
    [0, 0, 0, 1]
]) 

_C.CALIB.MATRIX_P2 = ([[719.787081,    0.,            608.463003, 44.9538775],
                [0.,            719.787081,    174.545111, 0.1066855],
                [0.,            0.,            1.,         3.0106472e-03],
                [0.,            0.,            0.,         0]])





def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()