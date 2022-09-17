from yacs.config import CfgNode as CN


_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.NUM_WORKERS = 4

_C.DATA = CN()
_C.DATA.DIR= "/data/kitti/3d_vision/training" 

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





def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()