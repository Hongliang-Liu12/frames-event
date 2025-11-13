import torch
import torch.nn as nn
import sys
import os

# --- 1. [!! 已修正 !!] 导入核心模块 ---
try:
    # 从 yolo 只导入 Head
    from .yolo import YOLOXHead 
    
    # 从 darknet 导入 Backbone 和所有基础模块
    from .darknet import CSPDarknet, BaseConv, CSPLayer, DWConv
    
except ImportError:
    # 修复作为独立脚本运行时可能出现的导入错误
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nets.yolo import YOLOXHead
    from nets.darknet import CSPDarknet, BaseConv, CSPLayer, DWConv


class SimpleTemporalNeck(nn.Module):
    """
    这个Neck只处理参考帧 (ref_frames)。
    它的职责是 "从过去的帧中，我学到了什么？"
    它 *绝对不会* 接触到 key_frame。
    """
    def __init__(self, in_channels, num_frame=3, act="silu"):
        super().__init__()
        n_ref = num_frame - 1  # 我们只关心 (num_frame - 1) 个参考帧
        
        # 1x1 卷积压缩，3x3 卷积学习局部
        self.conv_temporal = nn.Sequential(
            BaseConv(in_channels * n_ref, in_channels, 1, 1, act=act),
            BaseConv(in_channels, in_channels, 3, 1, act=act)
        )

    def forward(self, feats):
        # feats 列表: [frame1_feat, frame2_feat, key_frame_feat]
        
        # [!! 核心 !!] 我们只取参考帧
        ref_frames = feats[:-1]
        
        # 如果没有参考帧 (例如 num_frame=1), 返回空值或0
        if not ref_frames:
            # 这种情况在你的 num_frame=3 时不会发生
            # 但作为健壮性设计
            return torch.zeros_like(feats[0]) 

        r_feat_cat = torch.cat(ref_frames, dim=1)
        
        # 学习一个“时序上下文特征”并返回
        temporal_context = self.conv_temporal(r_feat_cat)
        return temporal_context



# --- 3. [!! 关键修正 !!] 内置的纯粹 YOLOX PAFPN 模块 ---
# 这个类是您 nets/yolo.py 中 YOLOPAFPN 的“精确副本”（只删除了 backbone）
class _BuiltIn_YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                 = DWConv if depthwise else BaseConv
        
        self.in_features    = in_features # 保留这个，即使没用到，以防万一
        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        # --- 所有层定义与您的 nets/yolo.py 完全一致 ---
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )
        self.bu_conv2       = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )
        self.bu_conv1       = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )

    def forward(self, input):
        feat1, feat2, feat3 = input

        # --- FPN 和 PAN 的剩余逻辑与您的 nets/yolo.py 完全一致 ---
        P5            = self.lateral_conv0(feat3)
        P5_upsample = self.upsample(P5)
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        P5_upsample = self.C3_p4(P5_upsample)

        P4            = self.reduce_conv1(P5_upsample) 
        P4_upsample = self.upsample(P4) 
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        P3_out      = self.C3_p3(P4_upsample)  

        P3_downsample   = self.bu_conv2(P3_out) 
        P3_downsample   = torch.cat([P3_downsample, P4], 1) 
        P4_out          = self.C3_n3(P3_downsample) 

        P4_downsample   = self.bu_conv1(P4_out)
        P4_downsample   = torch.cat([P4_downsample, P5], 1)
        P5_out          = self.C3_n4(P4_downsample)

        return (P3_out, P4_out, P5_out)


# --- 4. [!! 最终版 !!] YoloBodySST (V2 高效版) ---
class YoloBodySST(nn.Module):
    def __init__(self, num_classes, phi='s', num_frame=3):
        super().__init__()
        self.num_frame = num_frame
        
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False
        act = "silu"
        in_channels_config = [256, 512, 1024]
        
        # --- 架构拆分 (V2 核心) ---
        # 1. 骨干网络 (Backbone)
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        
        # 2. 时序融合颈 (Temporal Neck)
        c3_channels = int(in_channels_config[0] * width)
        c4_channels = int(in_channels_config[1] * width)
        c5_channels = int(in_channels_config[2] * width)

        self.neck_c3 = SimpleTemporalNeck(in_channels=c3_channels, num_frame=num_frame, act=act)
        self.neck_c4 = SimpleTemporalNeck(in_channels=c4_channels, num_frame=num_frame, act=act)
        self.neck_c5 = SimpleTemporalNeck(in_channels=c5_channels, num_frame=num_frame, act=act)


        # 3. 融合门 (Gating Mechanism)
        self.fusion_gate_c3 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.fusion_gate_c4 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.fusion_gate_c5 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

        self.attention_map_c3 = self.create_attention_map(c3_channels, act)
        self.attention_map_c4 = self.create_attention_map(c4_channels, act)
        self.attention_map_c5 = self.create_attention_map(c5_channels, act)

        # 4. 特征金字塔 (FPN)
        # [!! 关键 !!] 
        # 使用我们在这个文件里复制和修改的 _BuiltIn_YOLOPAFPN
        # 实例名 'fpn' 将 100% 匹配 "键名重命名" 逻辑
        self.fpn = _BuiltIn_YOLOPAFPN(depth, width, in_channels = in_channels_config, 
                                      depthwise = depthwise, act = act)
        
        # 5. 检测头 (Head)
        self.head = YOLOXHead(num_classes, width, in_channels=in_channels_config, 
                              depthwise=depthwise, act=act)


    def create_attention_map(self, in_channels, act):
            # 接收 C 通道特征, 输出 1 通道注意力图
            return nn.Sequential(
                BaseConv(in_channels, in_channels // 4, 3, 1, act=act),
                nn.Conv2d(in_channels // 4, 1, 1, 1, 0),
                nn.Sigmoid()
            )


    def forward(self, inputs):
        # inputs 维度: [bs, num_frame, c, h, w]
        
        # --- 1. 循环运行 Backbone ---
        c3_features, c4_features, c5_features = [], [], []
        for i in range(self.num_frame):
            backbone_outputs = self.backbone(inputs[:, i, :, :, :])
            c3 = backbone_outputs["dark3"] 
            c4 = backbone_outputs["dark4"]
            c5 = backbone_outputs["dark5"]
            c3_features.append(c3)
            c4_features.append(c4)
            c5_features.append(c5)
            
        # --- 2. 时序融合 ---
        fused_c3 = self.neck_c3(c3_features)
        fused_c4 = self.neck_c4(c4_features)
        fused_c5 = self.neck_c5(c5_features)
        
        # --- 3. 门控残差连接 ---
        original_c3 = c3_features[-1]
        original_c4 = c4_features[-1]
        original_c5 = c5_features[-1]


        # --- 4. [!! 关键 !!] 生成空间注意力图 ---
        # 根据"时序上下文"判断"在哪里"发生了运动
        attn_map_c3 = self.attention_map_c3(fused_c3) # [B, 1, H, W]
        attn_map_c4 = self.attention_map_c4(fused_c4)
        attn_map_c5 = self.attention_map_c5(fused_c5)

        # # --- 5. 融合 ---
        final_c3 = original_c3 + self.fusion_gate_c3 * attn_map_c3
        final_c4 = original_c4 + self.fusion_gate_c4 * attn_map_c4
        final_c5 = original_c5 + self.fusion_gate_c5 * attn_map_c5

        # --- 4. 运行 FPN (仅一次) ---
        fpn_inputs = (final_c3, final_c4, final_c5)
        p3, p4, p5 = self.fpn(fpn_inputs) # [!! 现在可以正常工作 !!]

        # --- 5. 运行 Head (仅一次) ---
        final_features = (p3, p4, p5)
        outputs = self.head(final_features)
        
        return outputs

# ==================================================== #
#                       测试代码
# ==================================================== #
if __name__ == "__main__":
    print("--- (最终修正版) 正在运行 YoloBodySST (V2 高效版) 模型完整性检查 ---")
    print("--- FPN 已内置 (使用 nets/yolo.py 的精确副本)，nets/yolo.py 未被修改 ---")

    num_classes = 10
    phi = 's'
    num_frame = 3
    
    print(f"正在实例化 YoloBodySST: num_classes={num_classes}, phi='{phi}', num_frame={num_frame}")
    
    try:
        model = YoloBodySST(num_classes=num_classes, phi=phi, num_frame=num_frame)
        model.eval()
        print("模型实例化成功。")

        # 打印新增的gate参数
        print(f"初始融合门 C3: {model.fusion_gate_c3.item()} (应为 0.0)")
        print(f"初始融合门 C4: {model.fusion_gate_c4.item()} (应为 0.0)")
        print(f"初始融合门 C5: {model.fusion_gate_c5.item()} (应为 0.0)")

        batch_size = 2
        height, width = 640, 640
        dummy_input = torch.randn(batch_size, num_frame, 3, height, width)
        print(f"\n创建虚拟输入张量，维度: {dummy_input.shape}")

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
            print(f"[失败] 模型输出格式不正确。")
            
    except ImportError as e:
        print(f"\n[错误] 导入失败，请确保 'nets.yolo' (YOLOXHead) 和 'nets.darknet' (CSPDarknet, BaseConv...) 存在: {e}")
    except Exception as e:
        print(f"\n[错误] 执行时发生错误: {e}")
        import traceback
        traceback.print_exc()