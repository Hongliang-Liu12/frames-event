import torch
import torch.nn as nn
import sys
import os

# --- 1. 导入项目核心模块 ---
try:
    # 尝试作为模块导入 (在你的项目中正常使用时)
    from .yolo import YOLOPAFPN, YOLOXHead
    from .darknet import BaseConv
except ImportError:
    # 尝试作为独立脚本运行时，修复导入路径
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nets.yolo import YOLOPAFPN, YOLOXHead
    from nets.darknet import BaseConv

# --- 2. 引入并优化 SSTNet 的时序融合颈 (MotionCouplingNeck) ---
class MotionCouplingNeck(nn.Module):
    """
    SSTNet中的运动耦合颈部模块，用于融合多帧特征。
    已修改为接受可变的输入通道数。
    """
    def __init__(self, in_channels, num_frame=3, act="silu"):
        super().__init__()
        self.num_frame = num_frame
        
        # 权重参数，用于加权每个参考帧与关键帧的融合结果
        self.weight = nn.ParameterList(
            nn.Parameter(torch.tensor([1.0 / (num_frame - 1)]), requires_grad=True) for _ in range(num_frame - 1)
        )
        
        # 定义网络层
        self.conv_ref = nn.Sequential(
            BaseConv(in_channels * (self.num_frame - 1), in_channels * 2, 3, 1, act=act),
            BaseConv(in_channels * 2, in_channels, 3, 1, act='sigmoid')
        )
        self.conv_cur = nn.Sequential(
            BaseConv(in_channels, in_channels, 3, 1, act=act), 
            BaseConv(in_channels, in_channels, 3, 1, act=act)
        )
        self.conv_cr_mix = nn.Sequential(
            BaseConv(in_channels * 2, in_channels * 2, 3, 1, act=act),
            BaseConv(in_channels * 2, in_channels, 3, 1, act=act)
        )
        self.conv_gl = nn.Sequential(
            BaseConv(in_channels * 2, in_channels * 2, 3, 1, act=act), 
            BaseConv(in_channels * 2, in_channels, 3, 1, act=act)
        )
        self.conv_gl_mix = nn.Sequential(
            BaseConv(in_channels, in_channels, 3, 1, act=act),
            BaseConv(in_channels, in_channels, 3, 1, act=act)
        )
        self.conv_final = nn.Sequential(
            BaseConv(in_channels * 2, in_channels * 2, 3, 1, act=act),
            BaseConv(in_channels * 2, in_channels, 3, 1, act=act)
        )

    def forward(self, feats):
        key_frame = feats[-1]
        ref_frames = feats[:-1]

        # 分支1
        r_feat = torch.cat(ref_frames, dim=1)
        r_feat = self.conv_ref(r_feat)       
        c_feat = self.conv_cur(r_feat * key_frame) 
        c_feat = self.conv_cr_mix(torch.cat([c_feat, key_frame], dim=1))
        
        # 分支2
        r_feats_list = [self.conv_gl(torch.cat([ref_frames[i], key_frame], dim=1)) * self.weight[i] for i in range(self.num_frame - 1)]
        r_feats_stack = torch.stack(r_feats_list, dim=0)
        r_feat_gl = self.conv_gl_mix(torch.sum(r_feats_stack, dim=0))
        
        # 最终融合
        final_fused_feat = self.conv_final(torch.cat([r_feat_gl, c_feat], dim=1))
        
        return final_fused_feat 

# --- 3. 构建基于“残差连接”策略的模型主体 ---
class YoloBodySST(nn.Module):
    def __init__(self, num_classes, phi='s', num_frame=3):
        super().__init__()
        self.num_frame = num_frame
        
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False
        act = "silu"

        self.backbone = YOLOPAFPN(depth, width, depthwise=depthwise, act=act)
        
        in_channels_config = [256, 512, 1024]
        p3_channels = int(in_channels_config[0] * width)
        p4_channels = int(in_channels_config[1] * width)
        p5_channels = int(in_channels_config[2] * width)

        self.neck_p3 = MotionCouplingNeck(in_channels=p3_channels, num_frame=num_frame, act=act)
        self.neck_p4 = MotionCouplingNeck(in_channels=p4_channels, num_frame=num_frame, act=act)
        self.neck_p5 = MotionCouplingNeck(in_channels=p5_channels, num_frame=num_frame, act=act)

        self.head = YOLOXHead(num_classes, width, in_channels=in_channels_config, depthwise=depthwise, act=act)
        
    def forward(self, inputs):
        # 1. 逐帧提取特征
        p3_features_over_time, p4_features_over_time, p5_features_over_time = [], [], []
        for i in range(self.num_frame):
            p3, p4, p5 = self.backbone(inputs[:, i, :, :, :])
            p3_features_over_time.append(p3)
            p4_features_over_time.append(p4)
            p5_features_over_time.append(p5)
            
        # 2. 时序融合
        fused_p3 = self.neck_p3(p3_features_over_time)
        fused_p4 = self.neck_p4(p4_features_over_time)
        fused_p5 = self.neck_p5(p5_features_over_time)
        
        # 3. !!! 关键不同点: 使用残差连接 (逐元素相加) !!!
        key_frame_p3 = p3_features_over_time[-1]
        key_frame_p4 = p4_features_over_time[-1]
        key_frame_p5 = p5_features_over_time[-1]

        final_p3 = key_frame_p3 + fused_p3
        final_p4 = key_frame_p4 + fused_p4
        final_p5 = key_frame_p5 + fused_p5
        
        # 4. 送入检测头
        final_features = (final_p3, final_p4, final_p5)
        outputs = self.head(final_features)
        
        return outputs

# ==================================================== #
#                       测试代码
# ==================================================== #
if __name__ == "__main__":
    print("--- 正在运行 YoloBodySST (残差版本) 模型完整性检查 ---")

    num_classes = 10
    phi = 's'
    num_frame = 3
    
    print(f"实例化模型: num_classes={num_classes}, phi='{phi}', num_frame={num_frame}")
    model = YoloBodySST(num_classes=num_classes, phi=phi, num_frame=num_frame)
    model.eval()
    print("模型实例化成功。")

    batch_size = 2
    height, width = 640, 640
    dummy_input = torch.randn(batch_size, num_frame, 3, height, width)
    print(f"\n创建虚拟输入张量，维度: {dummy_input.shape}")

    try:
        print("正在执行前向传播...")
        with torch.no_grad():
            outputs = model(dummy_input)
        print("前向传播完成。")

        print("\n--- 输出维度 (3个尺度) ---")
        if isinstance(outputs, (list, tuple)) and len(outputs) == 3:
            for i, out in enumerate(outputs):
                print(f"输出 P{i+3} 维度: {out.shape}")
            
            channels_head = num_classes + 5
            shape1_ok = list(outputs[0].shape) == [batch_size, channels_head, height//8, width//8]
            shape2_ok = list(outputs[1].shape) == [batch_size, channels_head, height//16, width//16]
            shape3_ok = list(outputs[2].shape) == [batch_size, channels_head, height//32, width//32]

            if shape1_ok and shape2_ok and shape3_ok:
                print("\n[成功] 模型输出维度正确！")
            else:
                print("\n[失败] 模型输出维度不正确。")
        else:
            print(f"[失败] 模型输出格式不正确，应为包含3个张量的元组。")

    except Exception as e:
        print(f"\n[错误] 前向传播时发生错误: {e}")
        import traceback
        traceback.print_exc()
