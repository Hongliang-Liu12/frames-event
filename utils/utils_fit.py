import torch
from tqdm import tqdm

from utils.utils import get_lr

   

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


    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f' % (loss / epoch_step))
    torch.save(model.state_dict(), 'log_frames/ep%03d-loss%.3f.pth' % (epoch + 1, loss / epoch_step))
