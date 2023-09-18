# OUTPUT_DIR = '../../semantic_kitti/checkpoints'
TRAIN_WHOLE_MODEL = True
# REAL
REAL_LOSS = True            # 是否使用real的cross entropy loss
CALIBRATION_LOSS = True     # 是否使用real的calibration loss
# PEBAL
ENERGY_LOSS = True          # 是否使用pebal的energy loss
GAMBLER_LOSS = True         # 是否使用pebal的gambler loss
GAUSSIAN_FILTER = True      # 是否使用pebal的gaussian filter

LR_DROP_STEP_SIZE = 10

# Shapenet Anomaly
SHAPENET_ANOMALY = False    # 是否将shapnet的物体作为训练时的异常

VAL_ENERGY_UNCERTAINTY = False  # 生成validation的输出时是否使用energy值作为uncertainty

SAVE_MODIFIED_DATA = False
SAVE_ALL_SEQUENCES = False
SAVE_PATH_DIR_FOR_MODIFIED_DATA = 'modified_dataset'