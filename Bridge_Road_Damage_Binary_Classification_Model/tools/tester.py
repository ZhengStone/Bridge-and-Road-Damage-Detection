import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader
from utils import MyDataset
from PIL import Image

__all__ = ['model_test']

def create_folder(path):
    try:
        # 检查路径是否存在，如果不存在则创建
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"文件夹 '{path}' 创建成功")
            return path
        else:
            print(f"文件夹 '{path}' 已存在")
            return path
    except Exception as e:
        print(f"创建文件夹 '{path}' 时出错：{e}")


def model_test(data_csv_path,
               model_,
               model_state_load_path,
               result_save_path,
               log_file_path,
               data_transform=None,
               data_target_transform=None,
               data_shuffle=False,
               data_num_workers=0,
                ):
    
    test_data = MyDataset(csv_path = data_csv_path, transform = data_transform, target_transform = data_target_transform)
    test_loader = DataLoader(dataset = test_data, batch_size = 1, shuffle = data_shuffle, num_workers = data_num_workers)

    model_.load_state_dict(torch.load(model_state_load_path))
    model_.eval()

    correct = 0.0
    total = 0.0
    label_list=[]
    output_list=[]

    for i, data_label in enumerate(test_loader):

        # 获取图片和标签
        inputs, labels = data_label
        inputs, labels = inputs.cuda(), labels.cuda()

        print(f"Processing folder: {i}")

        # forward
        outputs = model_(inputs)
        outputs.detach_()

        # 统计
        predicted = outputs.data > 0.5
        predicted = predicted.squeeze(dim=1)

        total += 1


        label_list.append(labels.cpu().numpy())
        output_list.append(outputs.cpu().numpy())

        # 统计混淆矩阵
        if predicted==True and labels==True:
            img_save_path = create_folder(os.path.join(result_save_path, 'TP'))
            test_transform = transforms.ToPILImage()
            test_img = test_transform(inputs[0])
            test_img.save(os.path.join(img_save_path, str(i)+'.jpg'))

            correct += 1
        
        if predicted==False and labels==False:
            img_save_path = create_folder(os.path.join(result_save_path, 'TN'))
            test_transform = transforms.ToPILImage()
            test_img = test_transform(inputs[0])
            test_img.save(os.path.join(img_save_path, str(i)+'.jpg'))

            correct += 1

        if predicted==True and labels==False:
            img_save_path = create_folder(os.path.join(result_save_path, 'FP'))
            test_transform = transforms.ToPILImage()
            test_img = test_transform(inputs[0])
            test_img.save(os.path.join(img_save_path, str(i)+'.jpg'))

        if predicted==False and labels==True:
            img_save_path = create_folder(os.path.join(result_save_path, 'FN'))
            test_transform = transforms.ToPILImage()
            test_img = test_transform(inputs[0])
            test_img.save(os.path.join(img_save_path, str(i)+'.jpg'))
        

    log_file_path = os.path.join(log_file_path, 'test_log.txt')
    with open(log_file_path, 'w') as log_file:
        test_info ="Accuracy:{:.2%}".format(correct / total)
        print(test_info)

        log_file.write(test_info + '\n \n')
        log_file.flush()
    
    output_list_np = np.array(output_list).squeeze()
    label_list_np = np.array(label_list).squeeze()

    np.save(os.path.join(result_save_path, 'output_list_np.npy'), output_list_np)
    np.save(os.path.join(result_save_path, 'label_list_np.npy'), label_list_np)

