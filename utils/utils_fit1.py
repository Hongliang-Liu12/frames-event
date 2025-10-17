import torch
from tqdm import tqdm

from utils.utils import get_lr
        
def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda):
    loss        = 0

    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = model_train(images)

            #----------------------#
            #   计算损失
            #----------------------#
            loss_value = yolo_loss(outputs, targets)

            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    model_train.eval()
    print('Start Validation')
    print('Finish Validation')
    
    loss_history.append_loss(loss / epoch_step)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f.pth' % (epoch + 1, loss / epoch_step))


# import torch
# from tqdm import tqdm

# from utils.utils import get_lr
# from eval import get_coco_map
# from eval import Evdet200kCocoDataset, letterbox_collate_fn
# from torch.utils.data import DataLoader



# # 1. 创建验证数据集和 COCO Ground Truth 对象
# DATASET_ROOT_DIR = "/home/lhl/Git/datasets/EvDET200K"
# BATCH_SIZE = 16

# # 真实值创建方式
# val_dataset = Evdet200kCocoDataset(DATASET_ROOT_DIR, split="test")
# val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=letterbox_collate_fn)
# coco_gt = val_dataset.coco # <--- 这就是我们需要的真实 coco_gt 对象

# def fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda,warmup_epochs=0,ema=None):
#     loss        = 0
#     val_loss    = 0

#     model_train.train()
#     print('Start Train')
#     with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',mininterval=0.3) as pbar:
#         for iteration, batch in enumerate(gen):
#             if iteration >= epoch_step:
#                 break
#             # --- Warmup 学习率调整 ---
#             if epoch < warmup_epochs:
#                 # 获取基础学习率
#                 base_lr = optimizer.param_groups[0]['lr']
#                 # 计算当前迭代的 warmup 学习率
#                 warmup_lr = base_lr * ((epoch * epoch_step + iteration) / (warmup_epochs * epoch_step))
#                 # 设置优化器学习率
#                 for param_group in optimizer.param_groups:
#                     param_group['lr'] = warmup_lr
#             # --------------------------            
#             images, targets = batch[0], batch[1]
#             with torch.no_grad():
#                 if cuda:
#                     images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
#                     targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]
#                 else:
#                     images  = torch.from_numpy(images).type(torch.FloatTensor)
#                     targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
#             #----------------------#
#             #   清零梯度
#             #----------------------#
#             optimizer.zero_grad()
#             #----------------------#
#             #   前向传播
#             #----------------------#
#             outputs         = model_train(images)

#             #----------------------#
#             #   计算损失
#             #----------------------#
#             loss_value = yolo_loss(outputs, targets)

#             #----------------------#
#             #   反向传播
#             #----------------------#
#             loss_value.backward()
#             optimizer.step()

#             if ema is not None:
#                 ema.update(model_train)

#             loss += loss_value.item()
            
#             pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
#                                 'lr'    : get_lr(optimizer)})
#             pbar.update(1)

#     print('Finish Train')
#     model_save = ema.ema if ema is not None else model
#     path = 'logs/ep%03d.pth' % (epoch + 1)
#     torch.save(model_save.state_dict(), path)
# # --- 验证与 COCO 评估阶段 (这是核心改动) ---
#     print('Start Validation')
#     # 确定用于评估的模型，优先使用 EMA
#     model_to_eval = ema.ema if ema is not None else model
    
#     # 获取设备信息
#     device = 'cuda' if cuda else 'cpu'
    
#     # 调用评估函数
#     coco_evaluator = get_coco_map(
#             model=model_to_eval,
#             dataloader=val_dataloader,
#             coco_gt=coco_gt, # <--- 使用从主脚本传入的真实 coco_gt
#             device=device,
#             # 你可以根据需要调整这里的参数
#             confidence=0.01,
#             nms_iou=0.65 
#         )
#     # 提取并打印 mAP 结果
#     val_map = 0.0
#     if coco_evaluator:
#         print("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35)
#         coco_evaluator.summarize()
#         # 提取关键指标: AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
#         val_map = coco_evaluator.stats[0] 
#         print(f"\nReturned mAP @[IoU=0.50:0.95]: {val_map:.4f}")
    
#     print('Finish Validation')
    
#     # --- 记录和保存 (使用 mAP 替代 val_loss) ---
#     # loss_history 现在记录的是 (train_loss, validation_mAP)
#     loss_history.append_loss(loss / epoch_step, val_map)
#     print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
#     print('Total Loss: %.3f || Val mAP: %.4f ' % (loss / epoch_step, val_map))

#     # 保存模型权重，文件名也反映 mAP
#     torch.save(model_to_eval.state_dict(), 'logs/ep%03d-loss%.3f-val_map%.4f.pth' % (epoch + 1, loss / epoch_step, val_map))