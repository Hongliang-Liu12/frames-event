#-------------------------------------#
#       完美解冻续训 (Unfreeze Perfect Resume)
#-------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from nets.yolo_frames_net import YoloBodySST
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory

import os
import sys
from loguru import logger # [!!] 导入 logger
import random
# 确保这里的导入路径与你的项目结构一致
from utils.frames_noaug_dataloader import EventJsonDataset,yolo_dataset_collate
from utils.utils_fit import fit_one_epoch
from utils.ema import ModelEMA
from frames_eval import (
    get_coco_map, 
    Evdet200kCocoDataset, 
    letterbox_collate_fn,
    print_per_class_results
)

CLASS_NAMES = [
    "people", "car", "bicycle", "electric bicycle", 
    "basketball", "ping_pong", "goose", "cat", "bird", "UAV"
]

# [!! 修改 1 !!] 
# get_optimizer_param_groups 已被移动到顶层 (全局范围)
# (您在您提供的代码中已经这样做了, 这是正确的)
def get_optimizer_param_groups(model_to_group):
    pg0, pg1, pg2 = [], [], [] 
    logger.info("Sorting parameters into optimizer groups...")               
    for k, v in model_to_group.named_parameters():
        if not v.requires_grad:
            continue  
        if k.endswith(".bias"):
            pg2.append(v)
        elif "fusion_gate" in k:
            pg2.append(v) 
        elif "bn" in k or k.endswith(".bn.weight"):
            pg0.append(v)
        else:
            pg1.append(v)
    
    logger.info(f"Optimizer groups: {len(pg0)} (no-decay BN), {len(pg1)} (decay weights), {len(pg2)} (no-decay biases/gates)")
    return pg0, pg1, pg2


if __name__ == "__main__":
    
    # ----------------------------------------------------#
    #    [!! 1. 超参数设置 !!]
    # ----------------------------------------------------#
    SEED = 42
    
    # [!! 修改 2 !!] 
    # Freeze_Epoch 必须与您 *保存* 的轮数 *之前* 的设置匹配
    # 您的路径是 '...ep006...' (第6轮), '42_6_...' (Freeze=6)
    # 这意味着您在第6轮(epoch=5)后保存的, 下一轮是 6
    # 因此, Freeze_Epoch 应该是 6 (代表 0,1,2,3,4,5 这6轮)
    # 您的原始脚本 'Freeze_Epoch=7' (0-6) 与 'ep006' 路径不匹配
    # 我将以您的 *路径* 为准, 假设 Freeze_Epoch=6
    Freeze_Epoch        = 7 # [!!] 关键: 必须匹配您保存时的设置 (例如 6 或 7)
    UnFreeze_Epoch      = 30
    Freeze_Lr           = 2e-4
    UnFreeze_Lr         = 1.5e-4
    
    # [!! 修改 3 !!] 
    # 建议为这个 "完美续训" 实验使用一个新目录
    log_dir = "logs/newtwostage/only2_perfect_resumeq" 
    os.makedirs(log_dir, exist_ok=True) 

    # ----------------------------------------------------#
    #    [!! 2. Loguru 设置 !!]
    # ----------------------------------------------------#
    
    # [!! 修改 4 !!] 
    # 使用更丰富的文件名前缀, 包含所有超参数, 以便跟踪
    file_prefix = f"{SEED}_{Freeze_Epoch}_{Freeze_Lr}_{UnFreeze_Lr}_6_n12ew"

# 2. 检查 log_dir 中是否有任何文件以此为前缀
    found_conflict = False
    conflicting_file = ""
    # (os.listdir 确保我们只检查此 log_dir)
    for filename in os.listdir(log_dir):
        if filename.startswith(file_prefix):
            found_conflict = True
            conflicting_file = filename
            break # 找到一个就足够了

    if found_conflict:
        # 此时 Logger 还未配置, 只能使用 print
        print(f"\n[!!] 错误: 实验配置冲突!")
        print(f"     检测到已存在使用相同前缀的日志文件:")
        print(f"     前缀: {file_prefix}")
        print(f"     文件: {os.path.join(log_dir, conflicting_file)}")
        print(f"     (这说明具有相同(SEED/Epoch/LR)参数的实验已在运行或已完成)")
        print(f"     请更换配置或清理日志目录。")
        print("     程序即将终止。")
        sys.exit(1) # 终止程序

    best_val_map = 0.0 
    initial_log_path = os.path.join(log_dir, f"{file_prefix}_{best_val_map:.4f}.txt")

    logger.remove() 
    logger.add(
        sys.stdout, 
        level="DEBUG", 
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    file_handler_id = logger.add(
        initial_log_path, 
        level="INFO", 
        encoding='utf-8', 
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )
    logger.info(f"--- [日志系统] ---")
    logger.info(f"    日志文件名前缀: {file_prefix}")
    logger.info(f"    初始日志文件: {initial_log_path}")
    logger.info(f"------------------")    
    
    # ... (SEED 设置, Cuda, device) ...
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED) 
    Cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and Cuda else 'cpu')
    
    #--------------------------------------------------------#
    #   设置训练集和测试集和预训练权重路径
    #--------------------------------------------------------#
    train_json_path = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/annotations/train.json'
    val_json_path   = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/annotations/test.json'
    image_root      = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/data'
    
    # [!! 修改 5 !!] 
    # 删除了 'model_path', 因为我们现在使用 A 和 B 路径
    # model_path      = '...' # (已移除)

    # [!!] 填入您刚刚保存的 "A" 模型的路径 [!!]
    resume_model_A_path = './logs/newtwostage/all2newstart1/42_12_0.0002_0.0001_ep007_model_map-0.5019.pth'
    # [!!] 填入您刚刚保存的 "B" 模型的路径 [!!]
    resume_model_B_path = './logs/newtwostage/all2newstart1/42_12_0.0002_0.0001_ep007_model_EMA_map-0.5019.pth'    

    # ... (input_shape, Cosine_scheduler, batch_size, num_workers) ...
    input_shape         = [640, 640]
    Cosine_scheduler    = True
    batch_size   = 8
    num_workers = 8
    num_classes = len(CLASS_NAMES)
    
    #===========================================================
    # eval: Setup validation dataloader (保持不变)
    #===========================================================
    print("Loading validation dataset for evaluation...")
    EVAL_DATASET_ROOT_DIR = "/home/lhl/Git/datasets/EvDET200K" 
    SEQ_LEN = 3
    EVAL_BATCH_SIZE = 4 
    val_dataset = Evdet200kCocoDataset(
        EVAL_DATASET_ROOT_DIR, 
        split="test", 
        seq_len=SEQ_LEN
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=EVAL_BATCH_SIZE, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True, 
        collate_fn=letterbox_collate_fn
    )
    coco_gt = val_dataset.coco 
    print("Validation dataset loaded.")
    
    #------------------------------------------------------#
    #   创建yolo模型
    #------------------------------------------------------#
    
    # [!! 修改 6 !!] 
    # 调整了 'model_train' 和 'ema_model' 的定义顺序
    # 确保它们在加载权重 *之前* 被创建
    
    model = YoloBodySST(num_classes,num_frame=3)
    model_train = model.train()
    
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = False
        cudnn.deterministic = True
        model_train = model_train.cuda()
        
    # EMA 模型 (B) 必须在加载前定义
    ema_model = ModelEMA(model_train, 0.9998)

    #------------------------------------------------------#
    #   [!!] 关键: 加载 "A" 和 "B" 的权重
    #------------------------------------------------------#    
    
    # (您原来的 'if model_path != ""' 逻辑已被替换)
    logger.info(f"--> [完美续训] 加载 'Model (A)' 权重: {os.path.basename(resume_model_A_path)}")
    model.load_state_dict(torch.load(resume_model_A_path, map_location=device))
    logger.info("    'Model (A)' 权重加载成功.")

    logger.info(f"--> [完美续训] 加载 'Model (B) EMA' 权重: {os.path.basename(resume_model_B_path)}")
    ema_model.ema.load_state_dict(torch.load(resume_model_B_path, map_location=device))
    logger.info("    'Model (B) EMA' 权重加载成功.")

    #------------------------------------------------------#
    #   [!! 修改 7 !!] 
    #   预评估 和 Loguru 日志更新
    #   (这替换了您原来的预评估逻辑)
    #------------------------------------------------------#
    logger.info('Start Validation (Pre-resume evaluation)')
    model_to_eval = ema_model.ema.eval() # [!!] 关键: 评估我们刚加载的 EMA (B) 模型
    coco_evaluator = get_coco_map(
        model=model_to_eval,
        dataloader=val_dataloader,
        coco_gt=coco_gt,
        device=device,
        confidence=0.01,
        nms_iou=0.65 
    )
    
    val_map_pre = 0.0 # 用于预评估的 mAP
    if coco_evaluator:
        logger.info("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35)
        coco_evaluator.summarize()
        val_map_pre = coco_evaluator.stats[0]
        val_map50 = coco_evaluator.stats[1]
        val_map75 = coco_evaluator.stats[2]
        logger.info(f" Average Precision (AP) @[ IoU=0.50:0.95 ] = {val_map_pre:.4f}")
        logger.info(f" Average Precision (AP) @[ IoU=0.50       ] = {val_map50:.4f}")
        logger.info(f" Average Precision (AP) @[ IoU=0.75       ] = {val_map75:.4f}")
        logger.info("="*70) 
        print_per_class_results(coco_evaluator, CLASS_NAMES)
        logger.info(f"\nReturned mAP @[IoU=0.50:0.95]: {val_map_pre:.4f}")
    
    logger.info('Finish Validation (Pre-resume evaluation)')

    # [!!] 关键: 更新 best_val_map 和日志文件名
    if val_map_pre > best_val_map:
        logger.info(f"\n[!!] 预评估发现新的最佳 mAP: {val_map_pre:.4f} (优于 {best_val_map:.4f})")
        old_log_path = os.path.join(log_dir, f"{file_prefix}_{best_val_map:.4f}.txt")
        new_log_path = os.path.join(log_dir, f"{file_prefix}_{val_map_pre:.4f}.txt")
        if os.path.exists(old_log_path):
            logger.info(f"    正在关闭: {os.path.basename(old_log_path)}")
            logger.remove(file_handler_id)
            try:
                os.rename(old_log_path, new_log_path)
                logger.info(f"    文件已重命名为: {os.path.basename(new_log_path)}")
            except Exception as e:
                logger.error(f"    [!!] 重命名日志文件失败: {e}")
            file_handler_id = logger.add(new_log_path, level="INFO", encoding='utf-8', format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")
        best_val_map = val_map_pre # [!!] 关键: 更新 mAP 跟踪器
    
    # --- 预评估结束 ---
    
    model_train = model.train() # 确保模型 (A) 返回训练模式
    yolo_loss    = YOLOLoss(num_classes,strides=[8, 16, 32])

#===========================================================
    # 开始训练
    #===========================================================
    
    # [!! 修改 8 !!] 
    # 'start_epoch' 必须从 'Freeze_Epoch' 开始
    start_epoch = Freeze_Epoch
    
    # [!! 修改 9 !!] 
    # 'end_epoch' 必须是总轮数
    end_epoch   = Freeze_Epoch + UnFreeze_Epoch
    
    # [!! 修改 10 !!] 
    # 'warmup_epochs' 设为 1 (或 0)
    # 这样 'epoch < warmup_epochs' (例如 6 < 1) 为 False, 从而 *禁用* 预热
    warmup_epochs=1 
    
    train_dataset   = EventJsonDataset(
        json_path=train_json_path,
        image_root=image_root,
        input_shape=input_shape,
        num_classes=num_classes
    )
    gen = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                     drop_last=True, collate_fn=yolo_dataset_collate)
    
    num_train = len(train_dataset)
    epoch_step      = num_train // batch_size

    optimizer    = None
    lr_scheduler = None
    
    # (ema_model 已经在加载前定义了)
    
    # [!! 修改 11 !!] 
    # 'ema_model.updates' 必须恢复到第 Freeze_Epoch 轮的状态
    ema_model.updates = epoch_step * start_epoch 
    logger.info(f"[!!] EMA 步数已设置为: {ema_model.updates} (模拟 {start_epoch} 轮训练)")

    # (val_map 在循环内被重置, 'best_val_map' 保存着预评估的值)
    
    for epoch in range(start_epoch, end_epoch):
        # [!! 修改 12 !!] 
        # 训练循环的 'if' 条件被修改
        # 原来是 'if epoch == 0:'
        # 现在是 'if epoch == Freeze_Epoch:'
        # 这样它只会在第一次循环时 (例如 epoch=6) 触发
        epoch_seed = SEED + epoch+3  # <--- [!!] 就是这行代码 [!!]
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        if epoch == Freeze_Epoch:
            # --------------------- #
            #     阶段 2: 解冻      #
            # --------------------- #
            logger.info("\n" + "="*70)
            logger.info(" " * 20 + f"Phase 2: Unfreeze Fine-tuning (Epochs {Freeze_Epoch} -> {end_epoch-1})")
            logger.info(f"Batch Size: {batch_size}, Learning Rate: {UnFreeze_Lr}")
            logger.info("Training: All layers.")
            logger.info("="*70 + "\n")

            # 1. 解冻所有层
            for param in model.parameters():
                param.requires_grad = True

            # 2. [!!] 创建一个 *全新* 的 "冷" 优化器 (100% 模拟)
            pg0, pg1, pg2 = get_optimizer_param_groups(model_train)
            optimizer = optim.Adam([
                {'params': pg0, 'weight_decay': 0.0},
                {'params': pg1, 'weight_decay': 5e-4},
                {'params': pg2, 'weight_decay': 0.0}
            ], lr=UnFreeze_Lr) 
            
            # 3. 为解冻阶段创建 *新* 的调度器
            if Cosine_scheduler:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UnFreeze_Epoch*epoch_step, eta_min=1e-5)
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
                                
        # --- 训练一个 Epoch (这行保持不变) ---
        # [!! 修改 13 !!] 
        # fit_one_epoch 的总轮数参数必须是 end_epoch
        fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch, 
                      epoch_step, gen, end_epoch, Cuda, ema_model, warmup_epochs=warmup_epochs, lr_scheduler=lr_scheduler, Cosine_scheduler=Cosine_scheduler)
        
        # ... (打印融合门, 保持不变) ...
        logger.info("\n--- 训练后融合门 (Fusion Gate) ---")
        logger.info(f"C3 门: {ema_model.ema.fusion_gate_c3.item():.4f}")
        logger.info(f"C4 门: {ema_model.ema.fusion_gate_c4.item():.4f}")
        logger.info(f"C5 门: {ema_model.ema.fusion_gate_c5.item():.4f}")
        logger.info("---------------------------------")
        
        # --- 评估和保存逻辑 (保持不变) ---
        val_map = 0.0 # 每次循环重置
        logger.info('Start Validation') 
        model_to_eval = ema_model.ema.eval() 
        coco_evaluator = get_coco_map(
            model=model_to_eval,
            dataloader=val_dataloader,
            coco_gt=coco_gt, 
            device=device,
            confidence=0.01,
            nms_iou=0.65 
        )
        
        if coco_evaluator:
            logger.info("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35) 
            coco_evaluator.summarize()
            val_map   = coco_evaluator.stats[0]
            val_map50 = coco_evaluator.stats[1]
            val_map75 = coco_evaluator.stats[2]
            logger.info("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35)
            logger.info(f" Average Precision (AP) @[ IoU=0.50:0.95 ] = {val_map:.4f}")
            logger.info(f" Average Precision (AP) @[ IoU=0.50       ] = {val_map50:.4f}")
            logger.info(f" Average Precision (AP) @[ IoU=0.75       ] = {val_map75:.4f}")
            logger.info("="*70)
            print_per_class_results(coco_evaluator, CLASS_NAMES)
            logger.info(f"\nReturned mAP @[IoU=0.50:0.95]: {val_map:.4f}") 
        logger.info('Finish Validation')

        # --- (日志重命名, 保持不变) ---
        if val_map > best_val_map:
            logger.info(f"\n[!!] Epoch {epoch+1} 发现新的最佳 mAP: {val_map:.4f} (优于 {best_val_map:.4f})")
            old_log_path = os.path.join(log_dir, f"{file_prefix}_{best_val_map:.4f}.txt")
            new_log_path = os.path.join(log_dir, f"{file_prefix}_{val_map:.4f}.txt")
            logger.info(f"     正在关闭: {os.path.basename(old_log_path)}")
            logger.remove(file_handler_id)
            try:
                os.rename(old_log_path, new_log_path)
                logger.info(f"     文件已重命名为: {os.path.basename(new_log_path)}")
            except Exception as e:
                logger.error(f"     [!!] 重命名日志文件失败: {e}")
            file_handler_id = logger.add(new_log_path, level="INFO", encoding='utf-8', format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")
            logger.info(f"     日志系统已更新，继续记录到新文件。")
            best_val_map = val_map
        else:
            logger.info(f"\n[i] Epoch {epoch+1} mAP 没有提升 (当前: {val_map:.4f}, 最佳: {best_val_map:.4f}). 日志文件名不变。")
        # --- 日志重命名逻辑结束 ---

        # [!! 修改 14 !!] 
        # 升级保存逻辑, 同时保存 A 和 B (使用 f-string)
        logger.info("Saving models for epoch %03d with mAP: %.4f" % (epoch + 1, val_map))

        # 1. 保存 "Model (A)" (即 'model', 无 .module)
        model_A_state = model.state_dict()
        save_A_path = os.path.join(log_dir, f"{file_prefix}_model_A_ep{epoch + 1:03d}_map-{val_map:.4f}.pth")
        torch.save(model_A_state, save_A_path)
        logger.info(f"    'Model (A)' saved to: {save_A_path}")

        # 2. 保存 "EMA (B)" (即 ema_model.ema)
        if ema_model:
            model_B_state = ema_model.ema.state_dict()
            save_B_path = os.path.join(log_dir, f"{file_prefix}_model_B_EMA_ep{epoch + 1:03d}_map-{val_map:.4f}.pth")
            torch.save(model_B_state, save_B_path)
            logger.info(f"    'Model (B) EMA' saved to: {save_B_path}")
        # --- 保存逻辑结束 ---