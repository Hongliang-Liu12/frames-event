#-------------------------------------#
#       对数据集进行训练
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

def get_optimizer_param_groups(model_to_group):
    pg0, pg1, pg2 = [], [], [] 
    logger.info("Sorting parameters into optimizer groups...")               
    for k, v in model_to_group.named_parameters():
        if not v.requires_grad:
            continue  # [!!] 只分组需要训练的参数
        if k.endswith(".bias"):
            pg2.append(v)
        elif "fusion_gate" in k:
            pg2.append(v) # fusion_gate 也不应有 weight decay
        elif "bn" in k or k.endswith(".bn.weight"):
            pg0.append(v)
        else:
            pg1.append(v)
    
    logger.info(f"Optimizer groups: {len(pg0)} (no-decay BN), {len(pg1)} (decay weights), {len(pg2)} (no-decay biases/gates)")
    return pg0, pg1, pg2


if __name__ == "__main__":
    
    # ----------------------------------------------------#
    #   [!! 1. 超参数设置 !!] 这一部分只用来进行非冻结训练
    # ----------------------------------------------------#
    SEED = 42
    Freeze_Epoch        = 7
    UnFreeze_Epoch      = 30
    Freeze_Lr           = 2e-4
    UnFreeze_Lr         = 2e-4
    
    # 定义日志目录
    log_dir = "logs/newtwostage/only2"
    os.makedirs(log_dir, exist_ok=True) # 确保这个目录存在

    # ----------------------------------------------------#
    #    [!! 2. Loguru 设置 !!]
    # ----------------------------------------------------#
    
    # A. 根据超参数创建文件名 "前缀"
    #    格式: 42_8_2e-4_2e-4
    file_prefix = f"{SEED}_{Freeze_Epoch}_{Freeze_Lr}_{UnFreeze_Lr}"
    
    # B. 初始化最佳 mAP 和 *初始* 日志文件名
    best_val_map = 0.0 # 初始 mAP 为 0.0
    initial_log_path = os.path.join(log_dir, f"{file_prefix}_{best_val_map:.4f}.txt")

    # C. [!! 关键 !!] 配置 loguru
    logger.remove() # 1. 移除 loguru 默认的 handler

    # 2. 添加控制台 handler (级别 DEBUG, 带颜色)
    logger.add(
        sys.stdout, 
        level="DEBUG", 
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    # 3. 添加初始文件 handler (级别 INFO)
    #    我们必须保存这个 handler_id 才能在后面移除它
    file_handler_id = logger.add(
        initial_log_path, 
        level="INFO", 
        encoding='utf-8',
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}" # 文件中的格式
    )

    # D. 使用 logger 打印启动信息
    logger.info(f"--- [日志系统] ---")
    logger.info(f"    日志文件名前缀: {file_prefix}")
    logger.info(f"    初始日志文件: {initial_log_path}")
    logger.info(f"------------------")    
    

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED) # 为所有GPU设置种子
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    Cuda                = True
    #--------------------------------------------------------#
    #   统一定义 device
    #--------------------------------------------------------#
    device = torch.device('cuda' if torch.cuda.is_available() and Cuda else 'cpu')
    #--------------------------------------------------------#
    #   设置训练集和测试集和预训练权重路径
    #--------------------------------------------------------#
    train_json_path = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/annotations/train.json'
    val_json_path   = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/annotations/test.json'
    image_root      = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/data'  # 图像根目录
    model_path      = '/home/lhl/Git/frames-event/logs/newtwostage/startmap0.505-0.793/ep007-map50_95-0.4978.pth'

    # [!!] 填入您刚刚保存的 "A" 模型的路径 [!!]
    resume_model_A_path = '/home/lhl/Git/frames-event/logs/newtwostage/all2/42_6_0.0002_0.0002_ep006_model_map-0.5005.pth' 
    
    # [!!] 填入您刚刚保存的 "B" 模型的路径 [!!]
    resume_model_B_path = '/home/lhl/Git/frames-event/logs/newtwostage/all2/42_6_0.0002_0.0002_ep006_model_EMA_map-0.5005.pth'    

    input_shape         = [640, 640]
    Cosine_scheduler    = True
    batch_size   = 8
    num_workers = 8
    num_classes = len(CLASS_NAMES)
    #===========================================================
    # eval: Setup validation dataloader (用于预训练评估和训练中评估)
    #===========================================================
    print("Loading validation dataset for evaluation...")
    EVAL_DATASET_ROOT_DIR = "/home/lhl/Git/datasets/EvDET200K" # 确保路径正确
    SEQ_LEN = 3
    EVAL_BATCH_SIZE = 4 # 你可以为验证设置一个单独的 batch_size
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
    coco_gt = val_dataset.coco # <-- 这就是你的 coco_gt
    print("Validation dataset loaded.")
    #------------------------------------------------------#
    #   创建yolo模型
    #------------------------------------------------------#
    model = YoloBodySST(num_classes,num_frame=3)
    model_train = model.train()
    #------------------------------------------------------#
    #   载入预训练权重
    #------------------------------------------------------#    
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = False
        cudnn.deterministic = True
        model_train = model_train.cuda()
    ema_model = ModelEMA(model_train, 0.9998)

    #------------------------------------------------------#
    #   [!!] 关键: 加载 "A" 和 "B" 的权重
    #------------------------------------------------------#    
    logger.info(f"--> Loading 'Model (A)' weights from: {resume_model_A_path}")
    model.load_state_dict(torch.load(resume_model_A_path, map_location=device))
    logger.info("    'Model (A)' weights loaded successfully.")

    logger.info(f"--> Loading 'Model (B) EMA' weights from: {resume_model_B_path}")
    ema_model.ema.load_state_dict(torch.load(resume_model_B_path, map_location=device))
    logger.info("    'Model (B) EMA' weights loaded successfully.")


    logger.info('Start Validation (Pre-training evaluation)')
    model_to_eval = ema_model.ema.eval() # [!!] 评估 EMA (B)  
    coco_evaluator = get_coco_map(
        model=model_to_eval,
        dataloader=val_dataloader,
        coco_gt=coco_gt,
        device=device,
        confidence=0.01,
        nms_iou=0.65 
    )
    
    val_map = 0.0
    if coco_evaluator:
        coco_evaluator.summarize()
        # 从 .stats 列表中提取您关心的三个值
        val_map   = coco_evaluator.stats[0]  # AP @[0.50:0.95]
        val_map50 = coco_evaluator.stats[1]  # AP @[0.50]
        val_map75 = coco_evaluator.stats[2]  # AP @[0.75]
        
        # 使用 logger.info() 将这三行写入日志文件
        logger.info("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35)
        logger.info(f" Average Precision (AP) @[ IoU=0.50:0.95 ] = {val_map:.4f}")
        logger.info(f" Average Precision (AP) @[ IoU=0.50       ] = {val_map50:.4f}")
        logger.info(f" Average Precision (AP) @[ IoU=0.75       ] = {val_map75:.4f}")
        logger.info("="*70) # 添加一个分隔符
        print_per_class_results(coco_evaluator, CLASS_NAMES)

        logger.info(f"\nReturned mAP @[IoU=0.50:0.95]: {val_map:.4f}") # <-- [修改]       
    logger.info('Finish Validation (Pre-training evaluation)')

    #------------------------------------------------------#
    #   准备训练
    #------------------------------------------------------#
    model_train = model.train() # 将模型设置回 .train() 模式
    
    # --- [!!! 修正结束：替换到这里为止 !!!] ---
    

    yolo_loss    = YOLOLoss(num_classes,strides=[8, 16, 32])

#===========================================================
    # 开始训练
    #===========================================================
    start_epoch = 0
    end_epoch   = UnFreeze_Epoch
    warmup_epochs=0
    
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

    # --- [!! 关键修正 !!] ---
    # 我们将 优化器 和 调度器 的 *定义* 移到循环内部
    # -------------------------
    optimizer    = None
    lr_scheduler = None

    # --- [OK] 用于 *训练* 的 EMA 初始化 (这部分是正确的)
    ema_model = ModelEMA(model_train, 0.9998)
    ema_model.updates = epoch_step * start_epoch 

    val_map = 0.0 # 在循环前初始化

    for epoch in range(start_epoch, end_epoch):
        # --- 阶段切换逻辑 (仅在 epoch 0 和 epoch Freeze_Epoch 时执行) ---
        if epoch == 0:
            # --- [!! 关键 !!] 定义参数分组的函数 ---
            # 每次创建优化器时，我们都需要重新运行这个逻辑

            # --- 参数分组函数定义结束 ---

            # --------------------- #
            #     阶段 2: 解冻      #
            # --------------------- #
            logger.info("\n" + "="*70)
            logger.info(" " * 20 + f"Phase 2: Unfreeze Fine-tuning (Epochs {end_epoch-1})")
            logger.info(f"Batch Size: {batch_size}, Learning Rate: {UnFreeze_Lr}")
            logger.info("Training: All layers.")
            logger.info("="*70 + "\n")

            # 1. 解冻所有层
            for param in model.parameters():
                param.requires_grad = True

            # 2. [!! 修正 !!] 再次重新创建 优化器
            # (这会捕获 *所有* 参数，并重置 momentum)
            pg0, pg1, pg2 = get_optimizer_param_groups(model_train)
            optimizer = optim.Adam([
                {'params': pg0, 'weight_decay': 0.0},
                {'params': pg1, 'weight_decay': 5e-4},
                {'params': pg2, 'weight_decay': 0.0}
            ], lr=UnFreeze_Lr) # [!!] 使用 UnFreeze_Lr
            # 3. 为解冻阶段创建 *新* 的调度器
            if Cosine_scheduler:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UnFreeze_Epoch*epoch_step, eta_min=1e-5)
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
                                
        # --- 训练一个 Epoch (这行保持不变) ---
        fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch, 
                      epoch_step, gen, UnFreeze_Epoch, Cuda, ema_model, warmup_epochs=warmup_epochs, lr_scheduler=lr_scheduler, Cosine_scheduler=Cosine_scheduler)
        

        # --- [!! 修正 !!] 打印融合门 (使用 logger.info) ---
        logger.info("\n--- 训练后融合门 (Fusion Gate) ---")
        # .module 是 DataParallel 包装器所必需的
        logger.info(f"C3 门: {ema_model.ema.fusion_gate_c3.item():.4f}")
        logger.info(f"C4 门: {ema_model.ema.fusion_gate_c4.item():.4f}")
        logger.info(f"C5 门: {ema_model.ema.fusion_gate_c5.item():.4f}")
        # print(f"C2 门: {model_train.module.fusion_gate_c2.item():.4f}") 
        logger.info("---------------------------------")
        # --- 打印结束 ---
        
        # --- 评估和保存逻辑 ---
        val_map = 0.0 # 每次循环重置
        
        logger.info('Start Validation') # <-- [修改]
        
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
            logger.info("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35) # <-- [修改]
            coco_evaluator.summarize()
            # 2. [!! 关键 !!] 从 .stats 列表中提取您关心的三个值
            val_map   = coco_evaluator.stats[0]  # AP @[0.50:0.95]
            val_map50 = coco_evaluator.stats[1]  # AP @[0.50]
            val_map75 = coco_evaluator.stats[2]  # AP @[0.75]
            
            # 3. [!! 关键 !!] 使用 logger.info() 将这三行写入日志文件
            logger.info("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35)
            logger.info(f" Average Precision (AP) @[ IoU=0.50:0.95 ] = {val_map:.4f}")
            logger.info(f" Average Precision (AP) @[ IoU=0.50       ] = {val_map50:.4f}")
            logger.info(f" Average Precision (AP) @[ IoU=0.75       ] = {val_map75:.4f}")
            logger.info("="*70) # 添加一个分隔符
            print_per_class_results(coco_evaluator, CLASS_NAMES)

            logger.info(f"\nReturned mAP @[IoU=0.50:0.95]: {val_map:.4f}") # <-- [修改]
        
        logger.info('Finish Validation') # <-- [修改]

        # --- 循环结束 ---

        # --- [!! 日志重命名逻辑 !!] ---
        # (这里是你粘贴 "步骤 3" 代码的地方)
        # (确保这个逻辑块在 `val_map` 被赋值之后,
        #  并且在下面的 "保存逻辑" 之前)
        if val_map > best_val_map:
            logger.info(f"\n[!!] Epoch {epoch+1} 发现新的最佳 mAP: {val_map:.4f} (优于 {best_val_map:.4f})")           
            # 1. 定义新旧路径
            old_log_path = os.path.join(log_dir, f"{file_prefix}_{best_val_map:.4f}.txt")
            new_log_path = os.path.join(log_dir, f"{file_prefix}_{val_map:.4f}.txt")           
            # 2. 移除旧 handler (关闭文件)
            logger.info(f"     正在关闭: {os.path.basename(old_log_path)}")
            logger.remove(file_handler_id)            
            # 3. 重命名文件
            try:
                os.rename(old_log_path, new_log_path)
                logger.info(f"     文件已重命名为: {os.path.basename(new_log_path)}")
            except Exception as e:
                logger.error(f"     [!!] 重命名日志文件失败: {e}")

            # 4. 添加新 handler (用新路径重新打开)
            file_handler_id = logger.add(new_log_path, level="INFO", encoding='utf-8', format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}")
            logger.info(f"     日志系统已更新，继续记录到新文件。")            
            # 5. 更新 mAP 跟踪器
            best_val_map = val_map
        else:
            logger.info(f"\n[i] Epoch {epoch+1} mAP 没有提升 (当前: {val_map:.4f}, 最佳: {best_val_map:.4f}). 日志文件名不变。")
        # --- [!! 日志重命名逻辑结束 !!] ---

        # --- [FIXED] 将保存逻辑移到评估块内部 ---
        if ema_model:
            save_model_state = ema_model.ema.state_dict()
            logger.info("Saving EMA model in train...epoch:%03d"% (epoch+1)) # <-- [修改]
        else:
            save_model_state = model.state_dict()
            logger.info("Saving raw model in train (EMA not enabled)...") # <-- [修改]
        logger.info("Saving model for epoch %03d with mAP: %.4f" % (epoch + 1, val_map))
        torch.save(save_model_state, 'logs/newtwostage/only2/ep%03d-map50_95-%.4f.pth' % (epoch + 1, val_map))
        logger.info("saved to: logs/newtwostage/only2/ep%03d-map50_95-%.4f.pth" % (epoch + 1, val_map))
        # --- 保存逻辑结束 ---