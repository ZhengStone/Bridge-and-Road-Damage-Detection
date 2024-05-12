# Bridge-and-Road-Damage-Detection-Using-Satellite-data-and-Deep-Learning-Models

The usage of train_run.py is as follows:
[!python3 train_run.py --train_data_csv_path "data/csv/train_data.csv" --train_data_batch_size 8 --train_max_epoch 100 --train_model_save_path "results/save_trained_model/effib1_fpn_convfc" --train_log_save_path "results/save_train_log/effib1_fpn_convfc"]

The usage of test_run.py is as follows:
[!python3 test_run.py --test_data_csv_path "data/csv/test_data.csv" --test_model_load_path "results/save_trained_model/effib1_fpn_convfc/model_params.pkl" --test_result_save_path "results/test_results_for_check/effib1_fpn_convfc" --test_log_save_path "results/test_results_for_check/effib1_fpn_convfc"]
