import torch
import torch.nn as nn
from model.main_model import res152_fpn_convfc_train, unet_iden_convfc_train, effib1_fpn_convfc_train, mobilev3_fpn_convfc_train 
from tools import model_train
import torchvision.transforms as transforms

import argparse

if __name__ == "__main__":
        
        parser = argparse.ArgumentParser(description='For model training.')
        parser.add_argument('--train_data_csv_path', type=str, help='Path to the CSV file containing training data.')
        parser.add_argument('--train_data_batch_size', type=int, help='Batch size for the training data.')
        parser.add_argument('--train_max_epoch', type=int, help='Maximum number of epochs for training.')
        parser.add_argument('--train_model_save_path', type=str, help='Path to save the trained model parameters.')
        parser.add_argument('--train_log_save_path', type=str, help='Path to save the training log file.')

        args = parser.parse_args()

        model_train(data_csv_path = args.train_data_csv_path,
                    data_batch_size = args.train_data_batch_size,
                    model_= effib1_fpn_convfc_train,
                    max_epoch = args.train_max_epoch,
                    model_save_path = args.train_model_save_path,
                    log_save_path = args.train_log_save_path,
                    data_transform = transforms.Compose([transforms.Pad((0, 0, 40, 160), fill=(0, 0, 0)),transforms.ToTensor()]),
                    data_target_transform = None,
                    data_shuffle = True,
                    data_num_workers = 0,
                    )