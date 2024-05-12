import torch
import torch.nn as nn
from model.main_model import res152_fpn_convfc_test, unet_iden_convfc_test, effib1_fpn_convfc_test, mobilev3_fpn_convfc_test 
from tools import model_test
import torchvision.transforms as transforms

import argparse

if __name__ == "__main__":
        
        parser = argparse.ArgumentParser(description='For model training.')
        parser.add_argument('--test_data_csv_path', type=str, help='Path to the CSV file containing testing data.')
        parser.add_argument('--test_model_load_path', type=str, help='Path to load the trained model parameters.')
        parser.add_argument('--test_result_save_path', type=str, help='The path where the test results will be saved.')
        parser.add_argument('--test_log_save_path', type=str, help='Path to save the testing log file.')

        args = parser.parse_args()
        
        model_test(data_csv_path = args.test_data_csv_path,
                   model_ = effib1_fpn_convfc_test,
                   model_state_load_path = args.test_model_load_path,
                   result_save_path = args.test_result_save_path,
                   log_file_path = args.test_log_save_path,
                   data_transform=transforms.Compose([transforms.Pad((0, 0, 40, 160), fill=(0, 0, 0)),transforms.ToTensor()]),
                   data_target_transform=None,
                   data_shuffle=False,
                   data_num_workers=0,
                   )