#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo_frames_net import YoloBodySST
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory

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
    yolox_pretrained_path ='/home/lhl/Git/YOLOX-main/YOLOX_outputs/evdet200k100/yolox_base/epoch_10_ckpt.pth'
    # yolox_pretrained_path=''
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
    Freeze_Epoch        = 10
    UnFreeze_Epoch      = 30
    
    #----------------------------------------------------#
    #   学习率超参数
    #   Freeze_Lr:   冻结阶段的学习率 (可以稍大)
    #   UnFreeze_Lr: 解冻阶段的学习率 (必须很小)
    #----------------------------------------------------#
    Freeze_Lr           = 1e-4
    UnFreeze_Lr         = 1e-5

    batch_size   = 4
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
        # device 已在顶部定义

        # 1. 加载预训练权重文件
        try:
            # [FIXED] 确保加载到正确的 device
            pretrained_state_dict = torch.load(yolox_pretrained_path, map_location=device)
        except Exception as e:
            print(f"Error loading weight file: {e}")

        # 2. [FIXED] 智能提取 state_dict，优先使用 'ema' 进行评估
        if 'ema' in pretrained_state_dict:
            print("    Checkpoint file detected. Found 'ema' state_dict, using it for evaluation.")
            pretrained_state_dict = pretrained_state_dict['ema']
        elif 'model' in pretrained_state_dict:
            print("    Checkpoint file detected. 'ema' not found, using 'model' state_dict.")
            pretrained_state_dict = pretrained_state_dict['model']
        elif 'state_dict' in pretrained_state_dict:
            print("    Checkpoint file detected. Found 'state_dict'.")
            pretrained_state_dict = pretrained_state_dict['state_dict']
        else:
            print("    No 'ema', 'model', or 'state_dict' key found. Assuming raw state_dict.")

        # 3. 获取当前模型的 state_dict
        model_state_dict = model.state_dict()
        
        # 4. 筛选出可以成功加载的权重
        load_dict = {
            k: v for k, v in pretrained_state_dict.items()
            if k in model_state_dict and model_state_dict[k].shape == v.shape
        }

        # 5. 打印加载信息
        model_keys = set(model_state_dict.keys())
        pretrained_keys = set(pretrained_state_dict.keys())
        loaded_keys = set(load_dict.keys())

        unloaded_keys = model_keys - loaded_keys
        
        print(f"    {len(loaded_keys)} out of {len(model_keys)} layers were successfully matched and prepared for loading.")
        
        if unloaded_keys:
            print(f"    The following {len(unloaded_keys)} layers were NOT found in the pretrained weights and will be trained from scratch:")
            for key in sorted(list(unloaded_keys))[:5]:
                print(f"      - {key}")
            if len(unloaded_keys) > 5:
                print("      - ... (and more)")

        # 6. 更新并加载权重
        model_state_dict.update(load_dict)
        model.load_state_dict(model_state_dict)
        print("    Weights loaded successfully.")

        # --- [FIXED] 评估加载好的 'model' ---
        print('Start Validation (Pre-training evaluation)')
        
        # [FIXED] 将模型移至 device 并设置为 .eval() 模式
        model.to(device)
        model_to_eval = model.eval() 
        
        # [FIXED] 删除了多余的 device 定义
        
        # 调用导入的评估函数
        coco_evaluator = get_coco_map(
            model=model_to_eval,
            dataloader=val_dataloader,
            coco_gt=coco_gt,
            device=device,
            confidence=0.01,
            nms_iou=0.65 
        )
        
        # 提取并打印 mAP 结果
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
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
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

    optimizer = optim.SGD(model_train.parameters(), lr=Freeze_Lr, momentum=0.9, weight_decay=5e-4)

    if Cosine_scheduler:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=end_epoch*epoch_step, eta_min=1e-5)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)


    # --- [OK] 用于 *训练* 的 EMA 初始化 (这部分是正确的)
    # 使用官方 YOLOX 的衰减率 0.9998
    # YOLOX 的 ModelEMA 类可以自动处理 DataParallel 包装器
    ema_model = ModelEMA(model_train, 0.9998)
    ema_model.updates = epoch_step * start_epoch # 如果从 0 开始, updates = 0

    # 在 epoch 循环开始前初始化 val_map，以防 epoch 0 不进行评估
    val_map = 0.0 

    for epoch in range(start_epoch, end_epoch):
        # --- 阶段切换逻辑 (仅在 epoch 0 和 epoch Freeze_Epoch 时执行) ---
        if epoch == 0:
            # --------------------- #
            #     阶段 1: 冻结      #
            # --------------------- #
            print("\n" + "="*70)
            print(" " * 20 + f"Phase 1: Freeze Training (Epochs 0 -> {Freeze_Epoch-1})")
            print(f"Batch Size: {batch_size}, Learning Rate: {Freeze_Lr}")
            print("Training: Only new temporal/fusion layers.")
            print("="*70 + "\n")
            
            # 1. 冻结层
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = False
            
            # 2. 为优化器设置正确的LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = Freeze_Lr

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

            # 2. 为优化器设置新的、更小的LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = UnFreeze_Lr

            # 3. 为解冻阶段创建 *新* 的调度器
            if Cosine_scheduler:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UnFreeze_Epoch*epoch_step, eta_min=1e-5)
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
                      
        fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch, 
                epoch_step, gen, Freeze_Epoch+UnFreeze_Epoch, Cuda,ema_model,warmup_epochs=warmup_epochs,lr_scheduler=lr_scheduler,Cosine_scheduler=Cosine_scheduler)
        print("--- 训练后的融合门 (Fusion Gate) ---")
        print(f"P3 Gate: {model.fusion_gate_p3.item()}")
        print(f"P4 Gate: {model.fusion_gate_p4.item()}")
        print(f"P5 Gate: {model.fusion_gate_p5.item()}")
        # --- [FIXED] 评估和保存逻辑 ---
        val_map = 0.0
        if epoch % 2 == 0 :
            print('Start Validation')
            model_to_eval = ema_model.ema.eval() # 训练期间评估 EMA
            # device 已在顶部定义
            
            coco_evaluator = get_coco_map(
               model=model_to_eval,
               dataloader=val_dataloader,
               coco_gt=coco_gt, 
               device=device,
               confidence=0.01,
               nms_iou=0.65 
            )
            
            # 提取并打印 mAP 结果
            if coco_evaluator:
               print("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35)
               coco_evaluator.summarize()
               
               print_per_class_results(coco_evaluator, CLASS_NAMES)
               
               val_map = coco_evaluator.stats[0] 
               print(f"\nReturned mAP @[IoU=0.50:0.95]: {val_map:.4f}")
            
            print('Finish Validation')

        # --- [FIXED] 将保存逻辑移到评估块内部 ---
        # 这样可以确保 val_map 是最新的
        if ema_model:
            save_model_state = ema_model.ema.state_dict()
            print("Saving EMA model in train...")
        else:
            save_model_state = model.state_dict()
            print("Saving raw model in train (EMA not enabled)...")
        torch.save(save_model_state, 'logs/twostage/startmap0.505-0.793/ep%03d-map50_95-%.4f.pth' % (epoch + 1, val_map))
        # --- 保存逻辑结束 ---