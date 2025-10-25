#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import os # [新增] 导入 os 模块用于创建文件夹

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
    yolox_pretrained_path ='/home/lhl/Git/YOLOX-main/YOLOX_outputs/evdet200k100/yolox_base/epoch_30_ckpt.pth'
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
    #----------------------------------------------------#
    Freeze_Epoch        = 10
    UnFreeze_Epoch      = 30
    
    #----------------------------------------------------#
    #   学习率超参数
    #----------------------------------------------------#
    Freeze_Lr           = 1e-4
    UnFreeze_Lr         = 1e-5

    batch_size   = 4
    num_workers = 8
    num_classes = len(CLASS_NAMES)

    #===========================================================
    # eval: Setup validation dataloader
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
    model = YoloBodySST(num_classes,num_frame=3)
    weights_init(model)
    #------------------------------------------------------#
    #   载入预训练权重 (这是我们修复好的版本)
    #------------------------------------------------------#    
    if yolox_pretrained_path != '':
        print(f'--> Loading pretrained YOLOX weights from: "{yolox_pretrained_path}"')
        
        try:
            pretrained_state_dict = torch.load(yolox_pretrained_path, map_location=device)
        except Exception as e:
            print(f"Error loading weight file: {e}")
            pretrained_state_dict = None

        if pretrained_state_dict:
            if 'model' in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict['model']
                print("    Checkpoint file detected. Extracted 'model' state_dict.")
            elif 'state_dict' in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict['state_dict']
                print("    Checkpoint file detected. Extracted 'state_dict'.")

            model_state_dict = model.state_dict()
            
            fpn_layer_prefixes = (
                'lateral_conv0', 'C3_p4', 'reduce_conv1', 'C3_p3',
                'bu_conv2', 'C3_n3', 'bu_conv1', 'C3_n4'
            )
            
            load_dict = {}
            print("    Mapping pretrained keys to new model structure...")
            
            for k, v in pretrained_state_dict.items():
                new_k = None
                
                if k.startswith('backbone.backbone.'):
                    new_k = k.replace('backbone.backbone.', 'backbone.', 1)
                elif k.startswith('backbone.') and any(k.startswith('backbone.' + p) for p in fpn_layer_prefixes):
                    new_k = 'fpn.' + k.replace('backbone.', '', 1)
                elif k.startswith(fpn_layer_prefixes):
                    new_k = 'fpn.' + k
                elif k.startswith('backbone.') or k.startswith('head.'):
                    new_k = k
                
                if new_k in model_state_dict:
                    if model_state_dict[new_k].shape == v.shape:
                        load_dict[new_k] = v
                    else:
                        print(f"  [跳过] 形状不匹配: {k} (预训练) -> {new_k} (当前)")

            model_keys = set(model_state_dict.keys())
            loaded_keys = set(load_dict.keys())
            unloaded_keys = model_keys - loaded_keys
            
            print(f"    {len(loaded_keys)} out of {len(model_keys)} layers were successfully matched and prepared for loading.")
            
            if unloaded_keys:
                print(f"    The following {len(unloaded_keys)} layers were NOT found in the pretrained weights or were mismatched, and will be trained from scratch:")
                unloaded_prefixes = set()
                for k in sorted(unloaded_keys):
                    prefix = ".".join(k.split('.')[:2])
                    unloaded_prefixes.add(prefix)
                
                for prefix in sorted(list(unloaded_prefixes)):
                    if prefix.startswith("motion_neck") or prefix.startswith("fusion_gate"):
                        print(f"      - {prefix}.* (新时序模块)")
                    elif not (prefix.startswith("backbone") or prefix.startswith("head") or prefix.startswith("fpn")):
                        print(f"      - {prefix}.*")
            
                if len(unloaded_keys) > 207: # 207 是新模块的层数
                    print("      - ... (以及未成功加载的 backbone/fpn 层)")


            model_state_dict.update(load_dict)
            model.load_state_dict(model_state_dict)
            print("    Weights loaded successfully.")

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
    
    # --- [FIXED] 你的两阶段训练逻辑从这里开始 ---

    # 暂时创建优化器和调度器 (将在 epoch 0 处被正确设置)
    optimizer = optim.SGD(model_train.parameters(), lr=Freeze_Lr, momentum=0.9, weight_decay=5e-4)
    if Cosine_scheduler:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=end_epoch*epoch_step, eta_min=1e-5)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    ema_model = ModelEMA(model_train, 0.9998)
    ema_model.updates = epoch_step * start_epoch 

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
            
            # 1. [FIXED] 冻结所有预训练层: backbone, fpn, head
            print("Freezing backbone, fpn, and head...")
            for param in model.backbone.parameters():
                param.requires_grad = False
            for param in model.fpn.parameters(): # <-- 修复Bug 1: 冻结 FPN
                param.requires_grad = False
            for param in model.head.parameters():
                param.requires_grad = False
            
            # 2. [FIXED] 重新创建优化器，使其只包含可训练的参数
            # (即 motion_neck 和 fusion_gate)
            params_to_train = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.SGD(params_to_train, lr=Freeze_Lr, momentum=0.9, weight_decay=5e-4)
            
            # 3. [FIXED] 为冻结阶段创建正确的调度器
            if Cosine_scheduler:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Freeze_Epoch*epoch_step, eta_min=Freeze_Lr * 0.01)
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
            print("Unfreezing all layers...")
            for param in model.parameters():
                param.requires_grad = True

            # 2. [FIXED] 重新创建优化器，使其包含 *所有* 参数
            # (这是修复 Bug 2 的关键)
            optimizer = optim.SGD(model.parameters(), lr=UnFreeze_Lr, momentum=0.9, weight_decay=5e-4)

            # 3. [FIXED] 为解冻阶段创建 *新* 的调度器
            if Cosine_scheduler:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=UnFreeze_Epoch*epoch_step, eta_min=UnFreeze_Lr * 0.01)
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
                      
        fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch, 
                epoch_step, gen, Freeze_Epoch+UnFreeze_Epoch, Cuda,ema_model,warmup_epochs=warmup_epochs,lr_scheduler=lr_scheduler,Cosine_scheduler=Cosine_scheduler)
        
        # [修改] 打印更易读的门控值
        print(f"Fusion Gate C3: {model.fusion_gate_c3.item():.4f}")
        print(f"Fusion Gate C4: {model.fusion_gate_c4.item():.4f}")
        print(f"Fusion Gate C5: {model.fusion_gate_c5.item():.4f}")
        
        # --- [FIXED] 评估和保存逻辑 ---
        # [修改] 每 2 个 epoch 评估一次
        if (epoch + 1) % 2 == 0:
            print(f'\n--- Start Validation Epoch {epoch+1} ---')
            model_to_eval = ema_model.ema.eval() # 训练期间评估 EMA
            
            coco_evaluator = get_coco_map(
               model=model_to_eval,
               dataloader=val_dataloader,
               coco_gt=coco_gt, 
               device=device,
               confidence=0.01,
               nms_iou=0.65 
            )
            
            # 提取并打印 mAP 结果
            val_map = 0.0 # 重置
            if coco_evaluator:
               print("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35)
               coco_evaluator.summarize()
               
               print_per_class_results(coco_evaluator, CLASS_NAMES)
               
               val_map = coco_evaluator.stats[0] 
               print(f"\nReturned mAP @[IoU=0.50:0.95]: {val_map:.4f}")
            
            print(f'--- Finish Validation Epoch {epoch+1} ---')

            # --- [FIXED] 将保存逻辑移到评估块内部 ---
            if ema_model:
                save_model_state = ema_model.ema.state_dict()
                print("Saving EMA model in train...")
            else:
                save_model_state = model.state_dict()
                print("Saving raw model in train (EMA not enabled)...")
            
            # [修改] 确保保存路径存在
            save_dir = 'logs/newtwostage/startmap0.505-0.793/'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(save_model_state, os.path.join(save_dir, 'ep%03d-map50_95-%.4f.pth' % (epoch + 1, val_map)))
            # --- 保存逻辑结束 ---