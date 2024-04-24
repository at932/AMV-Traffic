import time
import torchmetrics
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from utils.load_data import load_data
from utils.utils import trans_normalization,DeviceDataLoader,get_original_missing_mask,get_data_by_mask,get_data_by_mask_np
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error
from model.amvtn import A_MVGCN
import os

if __name__ == "__main__":
    dataset_name = 'metr_la'
    batch_size = 64
    index_id = 1
    input_dim = 1
    output_dim = 1

    if dataset_name == 'metr_la':
        input_dim = 1
        output_dim = 1
        seq_len = 12  # 1h
        hidden_dim = 64
        nodes_num = 207
        node_embed_dim = 16
        td_embed_dim = 288
        dw_embed_dim = 7
        envs_embed_dim = 9
        unknown_embed_dim = 16
        final_embed_dim = 10

    path = "./"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    miss_type = 'rm'
    miss_rate = 20

    print(miss_type, miss_rate)
    model_name = "amvtn"
    print(model_name)

    _, _, test_dataset, adj, max_val = load_data(path=path, index_id=index_id, miss_type=miss_type,
                                              miss_rate=miss_rate, model_name=model_name,
                                              dataset_name=dataset_name)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,shuffle=True)
    test_dataloader = DeviceDataLoader(test_dataloader, device)

    if model_name == "amvtn":
        model = A_MVGCN(input_dim=input_dim,
                        output_dim=output_dim,
                        node_num=nodes_num,
                        node_embed_dim=node_embed_dim,
                        td_embed_dim=td_embed_dim,
                        dw_embed_dim=dw_embed_dim,
                        envs_embed_dim=envs_embed_dim,
                        speed_past_dim=seq_len,
                        speed_future_dim=seq_len,
                        unknown_embed_dim=unknown_embed_dim,
                        final_embed_dim=final_embed_dim,
                        gcn_hidden_dim=hidden_dim)
    model.to(device)

    ckpt_path = path + "{}_{}{}.pt".format(model_name, miss_type, miss_rate, hidden_dim)
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt["model"])

    loss_fun = nn.MSELoss(reduction='mean')

    all_test_result, all_label, all_mask = None, None, None
    test_loss, test_mape, test_mae, test_rmse = 0, 0, 0, 0

    num_batches = len(test_dataloader)

    model.eval()  # 开启测试模型
    with torch.no_grad():
        for batch_idx, data in enumerate(test_dataloader):
            inputs, labels, masks, embed = data

            if model_name == "amvtn":
                pred = model(inputs, embed)

            pred = trans_normalization(pred, max_val)
            labels = trans_normalization(labels, max_val)

            # 计算精度，只用到缺失路段，但同时要把原始数据中就缺失的数据掩膜掉，两个步骤在一个函数中实现
            pred2, labels2 = get_data_by_mask(pred, labels, masks)

            mape = torchmetrics.functional.mean_absolute_percentage_error(pred2, labels2).item()
            mae = torchmetrics.functional.mean_absolute_error(pred2, labels2).item()
            rmse = torch.sqrt(torchmetrics.functional.mean_squared_error(pred2, labels2)).item()

            # test_loss += loss
            test_mape += mape
            test_mae += mae
            test_rmse += rmse

            pred0 = pred.cpu().detach().numpy()
            label0 = labels.cpu().detach().numpy()
            mask0 = masks.cpu().detach().numpy()
            if (batch_idx == 0):
                all_test_result = pred0
                all_label = label0
                all_mask = mask0
            else:
                all_test_result = np.concatenate((all_test_result, pred0), axis=0)
                all_label = np.concatenate((all_label, label0), axis=0)
                all_mask = np.concatenate((all_mask, mask0), axis=0)

        # test_loss /= num_batches
        test_mape /= num_batches
        test_mae /= num_batches
        test_rmse /= num_batches
        print('batch avg')
        print('test_rmse: %.4f | test_mae: %.4f | test_mape: %.4f'% (test_rmse, test_mae, test_mape))

    all_test_result2, all_label2 = get_data_by_mask_np(all_test_result, all_label, all_mask)
    all_mape = mean_absolute_percentage_error(all_label2, all_test_result2)
    all_mae = mean_absolute_error(all_label2, all_test_result2)
    all_rmse = np.sqrt(mean_squared_error(all_label2, all_test_result2))
    print('all avg')
    print('all_rmse: %.4f | all_mae: %.4f | all_mape: %.4f' % (all_rmse, all_mae, all_mape))







