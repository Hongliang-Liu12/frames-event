#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.yolo import YoloBody
from nets.yolo_training import YOLOLoss, weights_init
from utils.callbacks import LossHistory
from utils.dataloader import yolo_dataset_collate
# 确保这里的导入路径与你的项目结构一致
from utils.evnet_noaug import EventJsonDataset
from utils.utils_fit import fit_one_epoch


CLASS_NAMES = [
    "people", "car", "bicycle", "electric bicycle", 
    "basketball", "ping_pong", "goose", "cat", "bird", "UAV"
]


'''
训练自己的目标检测模型一定需要注意以下几点：
1、训练前仔细检查自己的格式是否满足要求，该库要求数据集格式为VOC格式，需要准备好的内容有输入图片和标签
   输入图片为.jpg图片，无需固定大小，传入训练前会自动进行resize。
   灰度图会自动转成RGB图片进行训练，无需自己修改。
   输入图片如果后缀非jpg，需要自己批量转成jpg后再开始训练。

   标签为.xml格式，文件中会有需要检测的目标信息，标签文件和输入图片文件相对应。

2、训练好的权值文件保存在logs文件夹中，每个epoch都会保存一次，如果只是训练了几个step是不会保存的，epoch和step的概念要捋清楚一下。
   在训练过程中，该代码并没有设定只保存最低损失的，因此按默认参数训练完会有100个权值，如果空间不够可以自行删除。
   这个并不是保存越少越好也不是保存越多越好，有人想要都保存、有人想只保存一点，为了满足大多数的需求，还是都保存可选择性高。

3、损失值的大小用于判断是否收敛，比较重要的是有收敛的趋势，即验证集损失不断下降，如果验证集损失基本上不改变的话，模型基本上就收敛了。
   损失值的具体大小并没有什么意义，大和小只在于损失的计算方式，并不是接近于0才好。如果想要让损失好看点，可以直接到对应的损失函数里面除上10000。
   训练过程中的损失值会保存在logs文件夹下的loss_%Y_%m_%d_%H_%M_%S文件夹中

4、调参是一门蛮重要的学问，没有什么参数是一定好的，现有的参数是我测试过可以正常训练的参数，因此我会建议用现有的参数。
   但是参数本身并不是绝对的，比如随着batch的增大学习率也可以增大，效果也会好一些；过深的网络不要用太大的学习率等等。
   这些都是经验上，只能靠各位同学多查询资料和自己试试了。
'''  
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

    # model_path      = '/home/lhl/Git/frames-event/utils/logs/ep030-loss2.922-val_loss4.231.pth'
    model_path      = ''
    #------------------------------------------------------#
    #   输入的shape大小，一定要是32的倍数
    #------------------------------------------------------#
    input_shape         = [640, 640]
    #------------------------------------------------------#
    #   所使用的YoloX的版本。nano、tiny、s、m、l、x
    #------------------------------------------------------#
    phi                 = 's'
    #------------------------------------------------------------------#
    #   YoloX的tricks应用
    #   Cosine_scheduler 余弦退火学习率 True or False
    #   由于数据增强已被移除，mosaic相关参数也一并删除
    #------------------------------------------------------------------#
    Cosine_scheduler    = True
    
    #----------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，batch_size最小为2，不能为1。
    #----------------------------------------------------#
    #----------------------------------------------------#
    #   冻结阶段训练参数
    #   此时模型的主干被冻结了，特征提取网络不发生改变
    #   占用的显存较小，仅对网络进行微调
    #----------------------------------------------------#
    Init_Epoch  = 0

    #----------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #----------------------------------------------------#
    End_Epoch   = 30
    batch_size  = 8
    lr          = 0.01 / 64.0 * batch_size
    #------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。

    #------------------------------------------------------#
    num_workers = 8

    #----------------------------------------------------#
    #   获取classes
    #----------------------------------------------------#
    num_classes = len(CLASS_NAMES)

    #------------------------------------------------------#
    #   创建yolo模型
    #------------------------------------------------------#
    model = YoloBody(num_classes, phi)

    weights_init(model)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    yolo_loss    = YOLOLoss(num_classes)
    loss_history = LossHistory("logs/")


    #===========================================================
    #开始训练
    start_epoch = 0
    end_epoch   = 60
    warmup_epochs=5
    optimizer = optim.SGD(model_train.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    # optimizer       = optim.Adam(model_train.parameters(), lr, weight_decay = 5e-4)
    if Cosine_scheduler:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=end_epoch, eta_min=1e-5)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

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

        
    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(model_train, model, yolo_loss, optimizer, epoch, 
                epoch_step, gen, end_epoch, Cuda,warmup_epochs=warmup_epochs)
        lr_scheduler.step()