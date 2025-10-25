#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from thop import profile, clever_format
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
    #   设置训练集和测试集的JSON文件路径
    #--------------------------------------------------------#
    train_json_path = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/annotations/train.json'
    val_json_path   = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/annotations/test.json'
    image_root      = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/data'  # 图像根目录
    # yolox_pretrained_path = '/home/lhl/Git/YOLOX-main/YOLOX_outputs/evdet200k100/yolox_base/epoch_83_ckpt.pth'
    # yolox_pretrained_path = '/home/lhl/Git/frames-event/logs/ep005-map50_950.5124.pth'
    # yolox_pretrained_path = '/home/lhl/Git/frames-event/logs/ep003-map50_950.4975.pth'
    # yolox_pretrained_path='/home/lhl/Git/frames-event/logs/twostage/startmap0.505-0.793/ep013-map50_95-0.5042.pth'
    yolox_pretrained_path=''
    model_path      = ''
    #------------------------------------------------------#
    #   输入的shape大小，一定要是32的倍数
    #------------------------------------------------------#
    input_shape         = [640, 640]

    #------------------------------------------------------------------#
    #   YoloX的tricks应用
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   由于数据增强已被移除，mosaic相关参数也一并删除
    #------------------------------------------------------------------#
    Cosine_scheduler    = True

    Init_Epoch  = 0

    End_Epoch   = 100
    batch_size  = 4
    lr          = 0.01 /64 *2
    num_workers = 8

    #----------------------------------------------------#
    #   获取classes
    #----------------------------------------------------#
    num_classes = len(CLASS_NAMES)

    #----------------------------------------------------#
    #   获取device
    #----------------------------------------------------#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #------------------------------------------------------#
    #   创建yolo模型
    #------------------------------------------------------#
    model = YoloBodySST(num_classes,num_frame=3)

    weights_init(model)

    #------------------------------------------------------#
    #   # 加载预训练权重文件
    #------------------------------------------------------#  

# ---------------- [ 把你原来的加载代码替换为这个 ] ----------------
    if yolox_pretrained_path != '':
        print(f'--> Loading pretrained YOLOX weights from: "{yolox_pretrained_path}"')
        
        # 1. 加载预训练权重文件
        try:
            pretrained_state_dict = torch.load(yolox_pretrained_path, map_location=device)
        except Exception as e:
            print(f"Error loading weight file: {e}")
            pretrained_state_dict = None

        if pretrained_state_dict:
            # 2. 如果是 checkpoint 文件，提取核心的 state_dict
            if 'model' in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict['model']
                print("    Checkpoint file detected. Extracted 'model' state_dict.")
            elif 'state_dict' in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict['state_dict']
                print("    Checkpoint file detected. Extracted 'state_dict'.")

            # 3. 获取当前模型的 state_dict
            model_state_dict = model.state_dict()
            
            # 4. [智能映射] 筛选并重命名权重
            
            # 官方YOLOX的FPN键
            fpn_layer_prefixes = (
                'lateral_conv0', 'C3_p4', 'reduce_conv1', 'C3_p3',
                'bu_conv2', 'C3_n3', 'bu_conv1', 'C3_n4'
            )
            
            load_dict = {}
            print("    Mapping pretrained keys to new model structure...")
            
            for k, v in pretrained_state_dict.items():
                new_k = None # 目标键名
                
                # 规则 1: 修复 'backbone.backbone.' -> 'backbone.'
                if k.startswith('backbone.backbone.'):
                    new_k = k.replace('backbone.backbone.', 'backbone.', 1)
                
                # 规则 2: 修复 'backbone.lateral...' -> 'fpn.lateral...'
                elif k.startswith('backbone.') and any(k.startswith('backbone.' + p) for p in fpn_layer_prefixes):
                    new_k = 'fpn.' + k.replace('backbone.', '', 1)

                # 规则 3: 修复 'lateral...' -> 'fpn.lateral...' (官方权重)
                elif k.startswith(fpn_layer_prefixes):
                    new_k = 'fpn.' + k

                # 规则 4: 保留 'backbone.stem...' 和 'head.stems...' (官方/标准权重)
                elif k.startswith('backbone.') or k.startswith('head.'):
                    new_k = k
                
                # 现在，使用你的简单逻辑来检查映射后的键
                if new_k in model_state_dict:
                    if model_state_dict[new_k].shape == v.shape:
                        load_dict[new_k] = v
                    else:
                        print(f"  [跳过] 形状不匹配: {k} (预训练) -> {new_k} (当前)")
                # else:
                    # 打印未被加载的键 (调试用)
                    # if new_k:
                    #     print(f"  [跳过] 键不匹配: {k} (预训练) -> {new_k} (未在当前模型中找到)")

            # 5. 打印加载信息 (你原来的代码)
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
            
                # (调试用) 打印未加载的预训练层
                if len(unloaded_keys) > (621-108): # 假设 108 是head, 621 是总数
                    print("      - ... (以及未成功加载的 backbone/fpn 层)")


            # 6. 更新并加载权重 (你原来的代码)
            model_state_dict.update(load_dict)
            model.load_state_dict(model_state_dict)
            print("    Weights loaded successfully.")

            # --- [FIXED] 评估加载好的 'model' ---
            print('Start Validation (Pre-training evaluation)')
            # ... (你原来的评估代码保持不变) ...
# ---------------- [ 替换结束 ] ----------------



    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # --- 计算模型参数量 ---
    # 在模型加载权重之后, DataParallel 包装之前
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\n" + "="*30 + " Model Stats " + "="*30)
    print(f"  Total parameters:     {total_params/1e6:.2f} M")
    print(f"  Trainable parameters: {trainable_params/1e6:.2f} M")
    print("="* (60 + len(" Model Stats ")))
    # -----------------------------
    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()



    #----------------------#
    yolo_loss    = YOLOLoss(num_classes,strides=[8, 16, 32])



    #===========================================================
    #eval daataset
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
    #===========================================================
    #开始训练
    start_epoch = 0
    end_epoch   = 200
    warmup_epochs=1
    # --- 修改这里：使用简化版的EventJsonDataset ---
    train_dataset   = EventJsonDataset(
        json_path=train_json_path,
        image_root=image_root,
        input_shape=input_shape,
        num_classes=num_classes
    )

    
    gen         = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                drop_last=True, collate_fn=yolo_dataset_collate)


    num_train = len(train_dataset)
    epoch_step      = num_train // batch_size
    optimizer = optim.SGD(model_train.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)



    if Cosine_scheduler:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=end_epoch*epoch_step, eta_min=1e-5)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

    #===========================================计算模型FLOPs和验证输出============================================
    print("\n" + "="*30 + " 开始验证 YoloBodySST 模型的前向传播和 FLOPs 计算 " + "="*30)
    # 2. 创建一个确定的虚拟输入
    model.eval()
    print("[验证步骤 2] 正在创建用于对比的虚拟输入数据...")
    # 使用 torch.ones 确保每次运行的输入都完全相同
    dummy_sequence_input = torch.ones(1, 3, 3, *input_shape).to(device)

    # --- [添加 FLOPs 计算模块] ---
    # 使用 thop 库和刚刚创建的虚拟输入
    print("\nCalculating Model FLOPs (using thop)...")
    # 将模型移到与输入相同的设备, 注意我们使用原始的 'model' (非 DataParallel)
    model_for_flops = model.to(device)
    # thop 需要一个元组/列表作为输入
    inputs_tuple = (dummy_sequence_input, )
 
    try:
        flops, params = profile(model_for_flops, inputs=inputs_tuple, verbose=False)
        # clever_format 自动转换为 G-FLOPs 和 M-Params
        flops_str, params_str = clever_format([flops, params], "%.2f")
        print(f"  Input shape: {list(dummy_sequence_input.shape)}")
        print(f"  FLOPs: {flops_str}")
        print(f"  Params (from thop): {params_str}")
    except Exception as e:
     print(f"  [ERROR] Error calculating FLOPs: {e}")
     print("  Skipping FLOPs calculation. (可能模型中有 thop 不支持的操作)")
    g = "  (Note: Params from thop should match 'Total parameters' printed earlier.)\n"

    model.to(device)
    # ---------------------------------
    # 3. 执行前向传播
    print("[验证步骤 3] 正在执行 YoloBodySST 的前向传播...")
    with torch.no_grad():
        output_sequence = model(dummy_sequence_input)
    
    # 4. 打印输出的一部分用于手动对比
    print("\n[验证步骤 4] 打印输出的一部分用于手动对比:")
    print("这是 YoloBodySST 输出的 P3 特征图 (第一个尺度) 的左上角 2x2 的值:")
    # 我们打印第一个批次, 第一个通道, H和W的前2个元素
    print(output_sequence[0][0, 0, :3, :3])
    


#===========================================开始测量FPS============================================
    print("\n" + "="*30 + " 开始测量模型推理速度 (FPS) " + "="*30)

    # 1. 确保模型在评估模式
    model.eval()
    
    # 2. 确保模型和输入在同一设备 (GPU)
    model.to(device)
    
    # 3. 创建一个虚拟输入 (与 FLOPs 计算时用的相同)
    #    我们使用 Batch Size = 1 来测量"延迟" (latency)
    #    使用 torch.randn 替代 torch.ones 来模拟更真实的数据
    dummy_input = torch.randn(1, 3, 3, *input_shape).to(device) 
    
    # 4. 预热 (Warm-up)
    #    在 GPU 上运行几次推理以确保 CUDA 核心被初始化, 避免启动开销影响计时
    print("  正在进行预热 (Warm-up runs)...")
    if Cuda:
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)

    # 5. 开始正式测量
    num_runs = 100
    print(f"  正在运行 {num_runs} 次推理来进行平均...")
    
    # --- 仅测量纯模型前向传播 ---
    # 使用 torch.cuda.Event 来精确计时 GPU 操作
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    # 记录开始时间
    starter.record() 
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            
    # 记录结束时间
    ender.record() 
    
    # 6. 计算时间
    torch.cuda.synchronize() # ！！关键：等待所有 GPU 异步操作完成
    
    total_time_ms = starter.elapsed_time(ender) # 毫秒
    total_time_s = total_time_ms / 1000.0       # 秒
    
    avg_time_per_run_s = total_time_s / num_runs
    fps = 1.0 / avg_time_per_run_s

    print("\n" + "="*25 + " 推理速度测试结果 " + "="*25)
    print(f"  测试硬件: {torch.cuda.get_device_name(device) if Cuda else 'CPU'}")
    print(f"  输入尺寸: {list(dummy_input.shape)}")
    print(f"  总共运行: {num_runs} 次")
    print(f"  总耗时: {total_time_s:.4f} 秒")
    print(f"  平均每帧耗时 (Latency): {avg_time_per_run_s * 1000:.4f} 毫秒 (ms)")
    print(f"  帧率 (FPS): {fps:.2f} FPS")
    print("=" * (50 + len(" 推理速度测试结果 ")))
    print("  注意: 这只是模型前向传播的速度，不包括数据预处理和NMS后处理。")
    #===========================================FPS测量结束============================================
    print("\n" + "="*30 + " 验证结束，程序将退出。 " + "="*30) 
    #===========================================计算模型FLOPs和验证输出============================================


    # --- [NEW] EMA 初始化 ---
    # 使用官方 YOLOX 的衰减率 0.9998
    ema_model = ModelEMA(model_train, 0.9998)
    ema_model.updates = epoch_step * start_epoch # 如果从 0 开始, updates = 0
    if yolox_pretrained_path != '' and model_path != '':
        print('Start First Validation')
        model_to_eval = ema_model.ema.eval()
        # 确定用于评估的模型 (model_train 是你的训练模型)
        # 如果你使用了 EMA (Exponential Moving Average)，你应该评估 EMA 模型
        # model_to_eval = model.eval() 
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 调用导入的评估函数
        # 注意: val_dataloader 和 coco_gt 是你在第 2 步中创建的
        print("--- 训练后的融合门 (Fusion Gate) ---")
        print(f"P3 Gate: {model.fusion_gate_p3.item()}")
        print(f"P4 Gate: {model.fusion_gate_p4.item()}")
        print(f"P5 Gate: {model.fusion_gate_p5.item()}")
        coco_evaluator = get_coco_map(
            model=model_to_eval,
            dataloader=val_dataloader,
            coco_gt=coco_gt, # <--- 使用从第 2 步传入的真实 coco_gt
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
            
            # (可选，但推荐) 打印逐类结果
            print_per_class_results(coco_evaluator, CLASS_NAMES)
            
            # 提取关键指标: AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
            val_map = coco_evaluator.stats[0] 
            print(f"\nReturned mAP @[IoU=0.50:0.95]: {val_map:.4f}")



    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch, 
                epoch_step, gen, end_epoch, Cuda,ema_model,warmup_epochs=warmup_epochs,lr_scheduler=lr_scheduler,Cosine_scheduler=Cosine_scheduler)

        # print("--- 训练后的融合门 (Fusion Gate) ---")
        # print(f"P3 Gate: {model.fusion_gate_p3.item()}")
        # print(f"P4 Gate: {model.fusion_gate_p4.item()}")
        # print(f"P5 Gate: {model.fusion_gate_p5.item()}")
        print(f"初始融合门 C3: {model.fusion_gate_c3.item()}")
        print(f"初始融合门 C4: {model.fusion_gate_c4.item()}")
        print(f"初始融合门 C5: {model.fusion_gate_c5.item()}")
        val_map = 0.0
        if epoch % 5 == 0 :
            print('Start Validation')
            model_to_eval = ema_model.ema.eval()
            # 确定用于评估的模型 (model_train 是你的训练模型)
            # 如果你使用了 EMA (Exponential Moving Average)，你应该评估 EMA 模型
            # model_to_eval = model.eval() 
            device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # 调用导入的评估函数
            # 注意: val_dataloader 和 coco_gt 是你在第 2 步中创建的

            # print("--- 训练后的融合门 (Fusion Gate) ---")
            # print(f"P3 Gate: {model.fusion_gate_p3.item()}")
            # print(f"P4 Gate: {model.fusion_gate_p4.item()}")
            # print(f"P5 Gate: {model.fusion_gate_p5.item()}")
            coco_evaluator = get_coco_map(
               model=model_to_eval,
               dataloader=val_dataloader,
               coco_gt=coco_gt, # <--- 使用从第 2 步传入的真实 coco_gt
               device=device,
               # 你可以根据需要调整这里的参数
               confidence=0.01,
               nms_iou=0.65 
            )
            
            # 提取并打印 mAP 结果
            
            if coco_evaluator:
               print("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35)
               coco_evaluator.summarize()
               
               # (可选，但推荐) 打印逐类结果
               print_per_class_results(coco_evaluator, CLASS_NAMES)
               
               # 提取关键指标: AP @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
               val_map = coco_evaluator.stats[0] 
               print(f"\nReturned mAP @[IoU=0.50:0.95]: {val_map:.4f}")
            
            print('Finish Validation')

        if ema_model:
            save_model_state = ema_model.ema.state_dict()
            print("Saving EMA model in train...")
        else:
            save_model_state = model.state_dict()
            print("Saving raw model in train (EMA not enabled)...")
        torch.save(save_model_state, 'logs/brandnewfrom0/ep%03d-map50_95-%.4f.pth' % (epoch + 1, val_map))
            
         


            