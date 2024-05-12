import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime
import os
from utils import MyDataset

__all__ = ['model_train']

def model_train(data_csv_path,
                data_batch_size,
                model_,
                max_epoch,
                model_save_path,
                log_save_path,
                data_transform=None,
                data_target_transform=None,
                data_shuffle=True,
                data_num_workers=0,
                ):
    
    train_data = MyDataset(csv_path = data_csv_path, transform = data_transform, target_transform = data_target_transform)
    train_loader = DataLoader(dataset = train_data, batch_size = data_batch_size, shuffle = data_shuffle, num_workers = data_num_workers)

    optimizer = optim.Adam(model_.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1) 

    # Open log file for writing
    log_file_path = os.path.join(log_save_path, 'train_log.txt')
    with open(log_file_path, 'w') as log_file:

        for epoch in range(max_epoch):

            start_time = datetime.now()

            loss_sigma = 0.0    # 记录一个epoch的loss之和
            correct = 0.0
            total = 0.0

            for i, data_label in enumerate(train_loader):
                # 获取图片和标签
                inputs, labels = data_label
                inputs, labels = inputs.cuda(), labels.cuda()
                labels = labels.unsqueeze(1).float()

                # forward, backward, update weights
                optimizer.zero_grad()
                outputs = model_(inputs)
                criterion = nn.BCELoss()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 统计预测信息
                predicted = outputs.data > 0.5
                total += labels.size(0)
                correct += (predicted == labels).squeeze().sum().cpu().numpy()
                loss_sigma += loss.item()

                # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
                if i % 10 == 9:
                    loss_avg = loss_sigma / 10
                    loss_sigma = 0.0
                    train_info ="Training: Epoch[{:0>3}/{:0>3}], Iteration[{:0>3}/{:0>3}], Loss_average:{:.4f}, Accuracy:{:.2%}, learning_rate:{}".format(
                        epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, correct / total, scheduler.get_last_lr())
                    print(train_info)

                    log_file.write(train_info + '\n \n')
                    log_file.flush()  # Ensure that the information is written immediately
                    
            scheduler.step()  # 更新学习率
            end_time = datetime.now()
            print('cost time:{}'.format(end_time-start_time))

        print('Finished Training')

    model_params_save_path = os.path.join(model_save_path, 'model_params.pkl')
    torch.save(model_.state_dict(), model_params_save_path)






