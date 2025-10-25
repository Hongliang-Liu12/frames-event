import torch
from tqdm import tqdm

from utils.utils import get_lr

   

def fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch, epoch_step, gen, Epoch, cuda,ema_model=None,warmup_epochs=0,lr_scheduler=None,Cosine_scheduler=False):
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
            # --- [NEW] 更新 EMA ---
            if ema_model:
                ema_model.update(model_train)
            if Cosine_scheduler and epoch >= warmup_epochs:
                lr_scheduler.step()

            # if iteration %1000 == 0:
            #     print('lr : %.6f' % get_lr(optimizer))
            loss += loss_value.item()
            
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)
            
        if not Cosine_scheduler and epoch >= warmup_epochs:
            lr_scheduler.step()
            print("step lr_scheduler")
    print('Finish Train')


    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f' % (loss / epoch_step))

    
# # --- [MODIFIED] 保存 EMA 模型的权重 ---
#     # 这样保存的 .pth 文件才是用于评估和推理的
#     if ema_model:
#         save_model_state = ema_model.ema.state_dict()
#         print("Saving EMA model state...")
#     else:
#         save_model_state = model.state_dict()
#         print("Saving raw model state (EMA not enabled)...")
#     torch.save(save_model_state, 'logs/brandnewfrom0/ep%03d-loss%.3f.pth' % (epoch + 1, loss / epoch_step))
