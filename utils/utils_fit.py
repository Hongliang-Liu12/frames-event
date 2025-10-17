import torch
from tqdm import tqdm

from utils.utils import get_lr
from eval import get_coco_map
from eval import Evdet200kCocoDataset, letterbox_collate_fn
from torch.utils.data import DataLoader



# 1. 创建验证数据集和 COCO Ground Truth 对象
DATASET_ROOT_DIR = "/home/lhl/Git/datasets/EvDET200K"
BATCH_SIZE = 16

# 真实值创建方式
val_dataset = Evdet200kCocoDataset(DATASET_ROOT_DIR, split="test")
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=letterbox_collate_fn)
coco_gt = val_dataset.coco # <--- 这就是我们需要的真实 coco_gt 对象        




def fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch, epoch_step, gen, Epoch, cuda,warmup_epochs=0):
    loss        = 0
    if epoch < warmup_epochs:
        # 在 Warmup 期间，我们需要一个固定的目标 LR 来计算线性增长。
        # 这里假设当前 epoch 开始时，optimizer.param_groups[0]['lr'] 是 Warmup 的目标 LR。
        target_lr = optimizer.param_groups[0]['lr']
    else:
        # 非 Warmup 阶段，LR 仅由外部调度器控制
        pass
    model_train.train()
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            # --- Warmup 学习率调整 (针对每个迭代) ---
            if epoch < warmup_epochs:
                total_warmup_steps = warmup_epochs * epoch_step
                current_step = epoch * epoch_step + iteration
                
                # 计算 Warmup 因子 (从 0 到 1 线性增长)
                warmup_factor = current_step / total_warmup_steps
                
                # 计算当前的 Warmup 学习率 (从 0 到 target_lr 线性增长)
                warmup_lr = target_lr * warmup_factor
                
                # 设置优化器学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            # ----------------------------------------
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
# --- 验证与 COCO 评估阶段 (这是核心改动) ---
    print('Start Validation')
    # 确定用于评估的模型，优先使用 EMA
    model_to_eval =model_train.eval()
    
    # 获取设备信息
    device = 'cuda' if cuda else 'cpu'
    
    # 调用评估函数
    coco_evaluator = get_coco_map(
            model=model_to_eval,
            dataloader=val_dataloader,
            coco_gt=coco_gt, # <--- 使用从主脚本传入的真实 coco_gt
            device=device,
            # 你可以根据需要调整这里的参数
            confidence=0.01,
            nms_iou=0.65 
        )
    # 提取并打印 mAP 结果
    val_map = 0.0
    if coco_evaluator:
        print("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35)
        coco_evaluator.summarize()
        # 提取关键指标: AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
        val_map = coco_evaluator.stats[0] 
        print(f"\nReturned mAP @[IoU=0.50:0.95]: {val_map:.4f}")
    
    print('Finish Validation')

    print('Finish Validation')

    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f' % (loss / epoch_step))
    torch.save(model.state_dict(), 'logs/ep%03d-loss%.3f.pth' % (epoch + 1, loss / epoch_step))
