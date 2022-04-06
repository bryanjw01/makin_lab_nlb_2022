import os
import unittest
from src.config import CHECKPOINT_PATH, LOG_PATH, RESULT_PATH, DATAPATH_DICT, MODEL_TYPE, DATASET_TYPE, MODEL, DATASET

class TestConfig(unittest.TestCase):
	
    def test_checkpoint_path(self):
        self.assertTrue(os.path.exists(CHECKPOINT_PATH), f'{RESULT_PATH} does not exist')
	
    def test_log_path(self):
        self.assertTrue(os.path.exists(LOG_PATH), f'{RESULT_PATH} does not exist')

    def test_result_path(self):
        self.assertTrue(os.path.exists(RESULT_PATH), f'{RESULT_PATH} does not exist')

    def test_data_path_dict(self):
        self.assertTrue(os.path.exists(DATAPATH_DICT[DATASET_TYPE.value]), f'{DATASET_TYPE.value} does not exist')

    def test_model_type(self):
        valid_type_enum = False
        if MODEL_TYPE == MODEL.RNN_F:
            valid_type_enum = True
        elif MODEL_TYPE == MODEL.NEURAL_ROBERTA:
            valid_type_enum = True
        elif MODEL_TYPE == MODEL.NEURAL_R_ROBERTA:
            valid_type_enum = True
        self.assertTrue(valid_type_enum, f'{MODEL_TYPE} not part of possible MODEL.')

    def test_dataset_type(self):
        valid_type_enum = False
        if DATASET_TYPE == DATASET.MC_MAZE:
            valid_type_enum = True
        elif DATASET_TYPE == DATASET.MC_RTT:
            valid_type_enum = True
        elif DATASET_TYPE == DATASET.AREA2_BUMP:
            valid_type_enum = True
        elif DATASET_TYPE == DATASET.DMFC:
            valid_type_enum = True
        elif DATASET_TYPE == DATASET.MC_MAZE_LARGE:
            valid_type_enum = True
        elif DATASET_TYPE == DATASET.MC_MAZE_MEDIUM:
            valid_type_enum = True
        elif DATASET_TYPE == DATASET.MC_MAZE_SMALL:
            valid_type_enum = True
        self.assertTrue(valid_type_enum, f'{DATASET_TYPE} not part of possible DATASETS.')

	
if __name__ == "__main__":
	unittest.main()