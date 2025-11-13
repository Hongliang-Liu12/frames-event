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

if __name__ == "__main__":
    SEED = 42  # 你可以选择任何固定的整数
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
    #   [FIXED] 统一定义 device
    #--------------------------------------------------------#
    device = torch.device('cuda' if torch.cuda.is_available() and Cuda else 'cpu')
    #--------------------------------------------------------#
    #   设置训练集和测试集和预训练权重路径
    #--------------------------------------------------------#
    train_json_path = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/annotations/train.json'
    val_json_path   = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/annotations/test.json'
    image_root      = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/data'  # 图像根目录
    # yolox_pretrained_path = '/home/lhl/Git/YOLOX-main/YOLOX_outputs/evdet200k100/yolox_base/epoch_30_ckpt.pth'
    # yolox_pretrained_path = '/home/lhl/Git/frames-event/logs/model/ep005-loss2.690-0.511.pth'
    yolox_pretrained_path ='/home/lhl/Git/YOLOX-main/YOLOX_outputs/evdet200k100/yolox_base/epoch_83_ckpt.pth'
    # yolox_pretrained_path=''
    # model_path      = '/home/lhl/Git/frames-event/logs/newtwostage/startmap0.505-0.793/ep007-map50_95-0.5214.pth'
    model_path      = ''

    #------------------------------------------------------#
    #   输入的shape大小，一定要是32的倍数
    #------------------------------------------------------#
    input_shape         = [640, 640]

    #------------------------------------------------------------------#
    #   Cosine_scheduler 余弦退火学习率 True or False
    #------------------------------------------------------------------#
    Cosine_scheduler    = True
    #----------------------------------------------------#
    #   两阶段训练超参数
    #   1. Freeze_Epoch:  冻结训练阶段的epoch数
    #   2. UnFreeze_Epoch: 解冻微调阶段的epoch数
    #   总训练epoch = Freeze_Epoch + UnFreeze_Epoch
    #----------------------------------------------------#
    Freeze_Epoch        = 8
    UnFreeze_Epoch      = 30
    
    #----------------------------------------------------#
    #   学习率超参数
    #   Freeze_Lr:   冻结阶段的学习率 (可以稍大)
    #   UnFreeze_Lr: 解冻阶段的学习率 (必须很小)
    #----------------------------------------------------#
    Freeze_Lr           = 2e-4
    UnFreeze_Lr         = 2e-4

    batch_size   = 8
    #----------------------------------------------------#
    #进程数
    #----------------------------------------------------#  
    num_workers = 8
    #----------------------------------------------------#
    #   获取classes
    #----------------------------------------------------#
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
    # 设定初始网络参数
    weights_init(model)
    #------------------------------------------------------#
    #   载入预训练权重
    #------------------------------------------------------#    

    # --- [FIXED] 预训练权重加载和评估逻辑 ---
    
    # [FIXED] 删除了此处错误的 EMA 初始化 (ema_model = ModelEMA(model_train, 0.9998))
    # [FIXED] 删除了 ema_model.updates = ...
    
    # [FIXED] 修改了 if 条件, 移除了 "and model_path != ''"
    if yolox_pretrained_path != '':
        print(f'--> Loading pretrained YOLOX weights from: "{yolox_pretrained_path}"')
        
        # 1. 加载预训练权重文件
        try:
            pretrained_state_dict = torch.load(yolox_pretrained_path, map_location=device)
        except Exception as e:
            print(f"Error loading weight file: {e}")
            pretrained_state_dict = None # 确保后续代码不会崩溃

        if pretrained_state_dict:
            # 2. 智能提取 state_dict
            if 'ema' in pretrained_state_dict:
                print("    Checkpoint file detected. Found 'ema' state_dict, using it.")
                pretrained_state_dict = pretrained_state_dict['ema']
            elif 'model' in pretrained_state_dict:
                print("    Checkpoint file detected. 'ema' not found, using 'model' state_dict.")
                pretrained_state_dict = pretrained_state_dict['model']
            elif 'state_dict' in pretrained_state_dict:
                print("    Checkpoint file detected. Found 'state_dict'.")
                pretrained_state_dict = pretrained_state_dict['state_dict']
            else:
                print("    No 'ema', 'model', or 'state_dict' key found. Assuming raw state_dict.")

            # --- 3. [!! 最终的键名重命名逻辑 !!] ---
            # 基于您的调试输出：
            # .pth 权重文件只有 'backbone' 和 'head' 两个顶层键
            
            print("    正在重命名预训练权重键名以匹配新架构...")
            massaged_state_dict = {}
            for k_pre, v_pre in pretrained_state_dict.items():
                new_key = None
                
                if k_pre.startswith('head.'):
                    # 规则 1: 'head.cls_preds...' -> 'head.cls_preds...'
                    # (Head 保持不变)
                    new_key = k_pre
                    
                elif k_pre.startswith('backbone.backbone.'):
                    # 规则 2: 'backbone.backbone.stem...' -> 'backbone.stem...'
                    # (将 V1 的 Backbone 键名 映射到 V2 的 Backbone 键名)
                    new_key = k_pre.replace('backbone.backbone.', 'backbone.', 1)
                    
                elif k_pre.startswith('backbone.'):
                    # 规则 3: 'backbone.C3_p4...' -> 'fpn.C3_p4...'
                    # (将 V1 的 FPN 键名 映射到 V2 的 FPN 键名)
                    # [!!] 这条规则必须在 "规则 2" 之后执行
                    new_key = k_pre.replace('backbone.', 'fpn.', 1)
                
                if new_key:
                    massaged_state_dict[new_key] = v_pre

            print(f"    键名重命名完成。")
            
            # 4. 获取当前模型的 state_dict
            model_state_dict = model.state_dict()
            
            # 5. 使用 "massaged_state_dict" (重命名后的) 进行筛选
            load_dict = {
                k: v for k, v in massaged_state_dict.items()
                if k in model_state_dict and model_state_dict[k].shape == v.shape
            }

            # 6. 打印加载信息 (这部分不变)
            model_keys_set = set(model_state_dict.keys())
            loaded_keys_set = set(load_dict.keys())
            unloaded_keys_set = model_keys_set - loaded_keys_set

            print(f"    {len(loaded_keys_set)} out of {len(model_keys_set)} (总模型层数) layers were successfully matched.")
            
            # 过滤掉我们 *期望* 未加载的新层
            expected_unloaded_prefixes = ('neck_c', 'fusion_gate_c')
            unexpected_unloaded_keys = []
            expected_unloaded_keys = []

            for key in unloaded_keys_set:
                if key.startswith(expected_unloaded_prefixes):
                    expected_unloaded_keys.append(key)
                else:
                    unexpected_unloaded_keys.append(key)

            if expected_unloaded_keys:
                print(f"    [信息] {len(expected_unloaded_keys)} 个新添加的层将从头训练 (这是正常的):")
                for key in sorted(list(expected_unloaded_keys))[:3]:
                    print(f"       - {key}")
                if len(expected_unloaded_keys) > 3: print("       - ... (and more)")
            
            if unexpected_unloaded_keys:
                print(f"    [!! 警告 !!] {len(unexpected_unloaded_keys)} 个 *非预期* 层未能加载 (这可能是个问题):")
                for key in sorted(unexpected_unloaded_keys)[:3]:
                    print(f"       - {key}")
                if len(unexpected_unloaded_keys) > 3: print("       - ... (and more)")

            # 7. 更新并加载权重
            model_state_dict.update(load_dict)
            model.load_state_dict(model_state_dict)
            print("    Weights loaded successfully.")


    if model_path != '':
        print(f'--> Loading model weights from: "{model_path}"')
        # 直接加载完整模型权重
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("    Model weights loaded successfully.")

    if yolox_pretrained_path != '' or model_path != '':
        # --- 评估加载好的 'model' ---
        print('Start Validation (Pre-training evaluation)')
        
        model.to(device)
        model_to_eval = model.eval() 
        
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
            print("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35)
            coco_evaluator.summarize()
            print_per_class_results(coco_evaluator, CLASS_NAMES)
            val_map = coco_evaluator.stats[0] 
            print(f"\nReturned mAP @[IoU=0.50:0.95]: {val_map:.4f}")

        print('Finish Validation (Pre-training evaluation)')
        # --- 预训练评估结束 ---

    #------------------------------------------------------#
    #   准备训练
    #------------------------------------------------------#
    model_train = model.train() # 将模型设置回 .train() 模式
    
    # --- [!!! 修正结束：替换到这里为止 !!!] ---
    
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = False
        cudnn.deterministic = True
        model_train = model_train.cuda()

    yolo_loss    = YOLOLoss(num_classes,strides=[8, 16, 32])
    loss_history = LossHistory("log_frames/")

#===========================================================
    # 开始训练
    #===========================================================
    start_epoch = 0
    end_epoch   = Freeze_Epoch + UnFreeze_Epoch
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
        
        if epoch == 0 or epoch == Freeze_Epoch:
            
            # --- [!! 关键 !!] 定义参数分组的函数 ---
            # 每次创建优化器时，我们都需要重新运行这个逻辑
            def get_optimizer_param_groups(model_to_group):
                pg0, pg1, pg2 = [], [], [] 
                print("Sorting parameters into optimizer groups...")
                
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
                
                print(f"Optimizer groups: {len(pg0)} (no-decay BN), {len(pg1)} (decay weights), {len(pg2)} (no-decay biases/gates)")
                return pg0, pg1, pg2
            # --- 参数分组函数定义结束 ---

            if epoch == 0:
                # --------------------- #
                #     阶段 1: 冻结      #
                # --------------------- #
                print("\n" + "="*70)
                print(" " * 20 + f"Phase 1: Freeze Training (Epochs 0 -> {Freeze_Epoch-1})")
                print(f"Batch Size: {batch_size}, Learning Rate: {Freeze_Lr}")
                print("Training: Head + All New Temporal Layers (Backbone/FPN frozen).")
                print("="*70 + "\n")
                
                # 1. 冻结层
                for name, param in model.named_parameters():
                    # 冻结 backbone 和 fpn
                    if name.startswith("backbone.") or name.startswith("fpn."):
                        param.requires_grad = False
                    # [!! 修正 !!] 确保 Head 和新层是可训练的
                    else:
                        param.requires_grad = True
                
                # 2. [!! 修正 !!] 重新创建 优化器
                # (这会捕获所有 requires_grad=True 的参数，并重置 momentum)
                pg0, pg1, pg2 = get_optimizer_param_groups(model_train)
                optimizer = optim.Adam([
                    {'params': pg0, 'weight_decay': 0.0},
                    {'params': pg1, 'weight_decay': 5e-4},
                    {'params': pg2, 'weight_decay': 0.0}
                ], lr=Freeze_Lr)
                # optimizer = optim.SGD([
                #     {'params': pg0, 'weight_decay': 0.0},
                #     {'params': pg1, 'weight_decay': 5e-4},
                #     {'params': pg2, 'weight_decay': 0.0}
                # ], lr=Freeze_Lr, momentum=0.9)

                # 3. 为冻结阶段创建调度器
                if Cosine_scheduler:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Freeze_Epoch*epoch_step, eta_min=1e-5)
                else:
                    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

            elif epoch == Freeze_Epoch:
                # --------------------- #
                #     阶段 2: 解冻      #
                # --------------------- #
                print("\n" + "="*70)
                print(" " * 20 + f"Phase 2: Unfreeze Fine-tuning (Epochs {Freeze_Epoch} -> {end_epoch-1})")
                print(f"Batch Size: {batch_size}, Learning Rate: {UnFreeze_Lr}")
                print("Training: All layers.")
                print("="*70 + "\n")

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
                # optimizer = optim.SGD([
                #     {'params': pg0, 'weight_decay': 0.0},
                #     {'params': pg1, 'weight_decay': 5e-4},
                #     {'params': pg2, 'weight_decay': 0.0}
                # ], lr=UnFreeze_Lr, momentum=0.9) # [!!] 使用 UnFreeze_Lr

                # 3. 为解冻阶段创建 *新* 的调度器
                if Cosine_scheduler:
                    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UnFreeze_Epoch*epoch_step, eta_min=1e-5)
                else:
                    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
                                
        # --- 训练一个 Epoch (这行保持不变) ---
        fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch, 
                      epoch_step, gen, Freeze_Epoch+UnFreeze_Epoch, Cuda, ema_model, warmup_epochs=warmup_epochs, lr_scheduler=lr_scheduler, Cosine_scheduler=Cosine_scheduler)
        

        # --- [!! 修正 !!] 打印融合门 (使用 .module 来安全访问) ---
        print("\n--- 训练后融合门 (Fusion Gate) ---")
            # .module 是 DataParallel 包装器所必需的
        print(f"C3 门: {ema_model.ema.fusion_gate_c3.item():.4f}")
        print(f"C4 门: {ema_model.ema.fusion_gate_c4.item():.4f}")
        print(f"C5 门: {ema_model.ema.fusion_gate_c5.item():.4f}")
        # print(f"C2 门: {model_train.module.fusion_gate_c2.item():.4f}") 
        print("---------------------------------")
        # --- 打印结束 ---
        # --- 评估和保存逻辑 (您的逻辑是正确的) ---
        val_map = 0.0 # 每次循环重置
        # if epoch % 2 == 0 or epoch>Freeze_Epoch: # 每2个 epoch 评估一次
        print('Start Validation')
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
            print("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35)
            coco_evaluator.summarize()
            print_per_class_results(coco_evaluator, CLASS_NAMES)
            val_map = coco_evaluator.stats[0] 
            print(f"\nReturned mAP @[IoU=0.50:0.95]: {val_map:.4f}")
        
        print('Finish Validation')

# --- 循环结束 ---

        # --- [FIXED] 将保存逻辑移到评估块内部 ---
        # 这样可以确保 val_map 是最新的
        if ema_model:
            save_model_state = ema_model.ema.state_dict()
            print("Saving EMA model in train...epoch:%03d"% (epoch))
        else:
            save_model_state = model.state_dict()
            print("Saving raw model in train (EMA not enabled)...")
        torch.save(save_model_state, 'logs/newtwostage/startmap0.505-0.793/ep%03d-map50_95-%.4f.pth' % (epoch + 1, val_map))
        # --- 保存逻辑结束 ---