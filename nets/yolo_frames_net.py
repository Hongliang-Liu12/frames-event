import torch
import torch.nn as nn

# --- 1. 从您原有的 "frames" 项目中导入核心模块 ---
# 我们使用 YOLOPAFPN 作为逐帧特征提取器，因为它能输出三个尺度
from .yolo import YOLOPAFPN, YOLOXHead 
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv

# --- 2. 引入 SSTNet 的时序融合颈 (Motion_coupling_Neck) ---
# （这个模块的 __init__ 已被修改，使其通道数可变）
class Motion_coupling_Neck(nn.Module):
    def __init__(self, channels=[128], num_frame=5):
        super().__init__()
        self.num_frame = num_frame
        
        # channels[0] 是一个整数，代表当前尺度的通道数
        # 我们用 'chan' 来存储它，以便在下面灵活使用
        chan = channels[0] 
        
        self.weight = nn.ParameterList(torch.nn.Parameter(torch.tensor([0.25]), requires_grad=True) for _ in range(num_frame))
        self.conv_ref = nn.Sequential(
            BaseConv(chan*(self.num_frame-1), chan*2, 3, 1),
            BaseConv(chan*2, chan, 3, 1, act='sigmoid')
        )
        self.conv_cur = nn.Sequential(BaseConv(chan, chan, 3, 1), BaseConv(chan, chan, 3, 1))
        self.conv_gl = nn.Sequential(BaseConv(chan*2, chan*2, 3, 1), BaseConv(chan*2, chan, 3, 1))
        self.conv_gl_mix = nn.Sequential(BaseConv(chan, chan, 3, 1), BaseConv(chan, chan, 3, 1))
        self.conv_cr_mix = nn.Sequential(BaseConv(chan*2, chan*2, 3, 1), BaseConv(chan*2, chan, 3, 1))
        self.conv_final = nn.Sequential(BaseConv(chan*2, chan*2, 3, 1), BaseConv(chan*2, chan, 3, 1))

    def forward(self, feats):
        # feats 是一个列表, 包含 [frame0, frame1, ..., frame_key] 的特征
        r_feat = torch.cat([feats[j] for j in range(self.num_frame-1)], dim=1)
        r_feat = self.conv_ref(r_feat)
        c_feat = self.conv_cur(r_feat*feats[-1])
        c_feat = self.conv_cr_mix(torch.cat([c_feat, feats[-1]], dim=1))
        r_feats = torch.stack([self.conv_gl(torch.cat([feats[i], feats[-1]], dim=1))*self.weight[i] for i in range(self.num_frame-1)], dim=0)
        r_feat = self.conv_gl_mix(torch.sum(r_feats, dim=0))
        c_feat = self.conv_final(torch.cat([r_feat, c_feat], dim=1))
        # Motion_coupling_Neck的输出是一个列表，我们取出第0个元素
        return c_feat 

# --- 3. 构建新的“多尺度-多帧”模型主干 ---
class YoloBodySST(nn.Module):
    def __init__(self, num_classes, phi='s', num_frame=3): # 我们重新引入 phi 参数
        super().__init__()
        self.num_frame = num_frame
        
        # 定义 YOLOX 的标准尺寸
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False

        # 模块一: 特征提取器 (使用 YOLOPAFPN)
        # YOLOPAFPN 将作为我们“逐帧”的特征提取器
        self.backbone = YOLOPAFPN(depth, width, depthwise=depthwise)
        
        # 模块二: 时序融合颈 (为 P3, P4, P5 分别创建一个)
        # 计算 P3, P4, P5 的输出通道数
        in_channels_config = [256, 512, 1024]
        p3_channels = int(in_channels_config[0] * width) # e.g., 's' (width=0.50) -> 128
        p4_channels = int(in_channels_config[1] * width) # e.g., 's' (width=0.50) -> 256
        p5_channels = int(in_channels_config[2] * width) # e.g., 's' (width=0.50) -> 512

        self.neck_p3 = Motion_coupling_Neck(channels=[p3_channels], num_frame=num_frame)
        self.neck_p4 = Motion_coupling_Neck(channels=[p4_channels], num_frame=num_frame)
        self.neck_p5 = Motion_coupling_Neck(channels=[p5_channels], num_frame=num_frame)

        self.adapter_p3 = BaseConv(p3_channels * 2, p3_channels, 1, 1, act="silu")
        self.adapter_p4 = BaseConv(p4_channels * 2, p4_channels, 1, 1, act="silu")
        self.adapter_p5 = BaseConv(p5_channels * 2, p5_channels, 1, 1, act="silu")
        # 模块三: 检测头 (使用 YOLOXHead)
        # 它天生就能处理 [P3, P4, P5] 的输入
        self.head = YOLOXHead(num_classes, width, in_channels=in_channels_config, depthwise=depthwise)
        
    def forward(self, inputs):
        # inputs 维度: [bs, nums, c, h, w]
        
        # 1. 逐帧提取 P3, P4, P5 特征
        p3_features = []
        p4_features = []
        p5_features = []


        for i in range(self.num_frame):
            # p3, p4, p5 是 YOLOPAFPN 的三个尺度输出
            p3, p4, p5 = self.backbone(inputs[:, i, :, :, :])
            p3_features.append(p3)
            p4_features.append(p4)
            p5_features.append(p5)
            
        # # 2. 在每个尺度上分别进行时序融合
        # fused_p3 = self.neck_p3(p3_features)
        # fused_p4 = self.neck_p4(p4_features)
        # fused_p5 = self.neck_p5(p5_features)
        
        # 假设关键帧是最后一帧 (索引 -1)
        original_p3 = p3_features[-1]
        original_p4 = p4_features[-1]
        original_p5 = p5_features[-1]
        # ----------------------------------------------------

        # 2. 在每个尺度上分别进行时序融合
        fused_p3 = self.neck_p3(p3_features)
        fused_p4 = self.neck_p4(p4_features)
        fused_p5 = self.neck_p5(p5_features)

        # --- !!! 关键修改点 3: 拼接原始特征和融合特征 !!! ---
        concat_p3 = torch.cat([original_p3, fused_p3], dim=1) # 通道维度拼接
        concat_p4 = torch.cat([original_p4, fused_p4], dim=1)
        concat_p5 = torch.cat([original_p5, fused_p5], dim=1)
        # -----------------------------------------------------

        # --- !!! 关键修改点 4: 通过适配器调整通道数 !!! ---
        adapted_p3 = self.adapter_p3(concat_p3)
        adapted_p4 = self.adapter_p4(concat_p4)
        adapted_p5 = self.adapter_p5(concat_p5)        
        # 3. 将融合后的三个尺度特征送入检测头
        # YOLOXHead 期望一个包含3个张量的元组 (P3, P4, P5)
        # fused_features = (fused_p3, fused_p4, fused_p5)
        fused_features = (adapted_p3, adapted_p4, adapted_p5)
        outputs = self.head(fused_features)
        
        return outputs

# ==================================================== #
#                  测试代码 (main 函数)
# ==================================================== #
if __name__ == "__main__":
    # --- 添加这三行来修复相对导入错误 ---
    import sys
    import os
    # 将项目的根目录 (nets 文件夹的上一级) 添加到 sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # --------------------------------------------------

    print("--- 正在运行 多尺度SSTNet模型 完整性检查 ---")

    # 1. 定义模型参数
    num_classes = 10    # 假设10个类别
    phi = 's'           # 使用 's' 尺寸 (width=0.50)
    num_frame = 3      # 输入的序列长度 (与您代码一致)
    
    # 2. 实例化新模型
    print(f"正在实例化 YoloBodySST (多尺度版本)...")
    print(f"参数: num_classes={num_classes}, phi='{phi}', num_frame={num_frame}")
    model = YoloBodySST(num_classes=num_classes, phi=phi, num_frame=num_frame)
    model.eval()
    print("模型实例化成功。")

    # 3. 创建一个虚拟输入张量
    batch_size = 2
    height, width = 640, 640
    # 输入维度: [bs, nums, c, h, w]
    dummy_input = torch.randn(batch_size, num_frame, 3, height, width)
    print(f"\n创建虚拟输入张量，维度: {dummy_input.shape}")

    # 4. 执行前向传播
    try:
        print("正在执行前向传播...")
        with torch.no_grad():
            outputs = model(dummy_input)
        print("前向传播完成。")

        # 5. 打印输出维度
        print("\n--- 输出维度 (应为 3 个尺度) ---")
        if isinstance(outputs, (list, tuple)):
            for i, out in enumerate(outputs):
                print(f"输出 [{i}] 维度: {out.shape}")
        else:
            print(f"输出维度: {outputs.shape}")
        
        # 验证输出
        channels_head = num_classes + 5 # 10个类别 + 5 (xywh, obj)
        
        print("\n期望的输出维度:")
        print(f"[0]: [{batch_size}, {channels_head}, {height//8}, {width//8}]  (80x80)")
        print(f"[1]: [{batch_size}, {channels_head}, {height//16}, {width//16}] (40x40)")
        print(f"[2]: [{batch_size}, {channels_head}, {height//32}, {width//32}] (20x20)")

        # 检查是否满足所有条件
        shape1_correct = list(outputs[0].shape) == [batch_size, channels_head, height//8, width//8]
        shape2_correct = list(outputs[1].shape) == [batch_size, channels_head, height//16, width//16]
        shape3_correct = list(outputs[2].shape) == [batch_size, channels_head, height//32, width//32]

        if len(outputs) == 3 and shape1_correct and shape2_correct and shape3_correct:
             print("\n[成功] 模型输出维度正确 (3个尺度)！")
        else:
             print("\n[失败] 模型输出维度不正确。")

    except Exception as e:
        print(f"\n[错误] 前向传播时发生错误: {e}")
        import traceback
        traceback.print_exc()