import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import time

# --- 1. 从 YOLOX 源代码中导入必要的模块 ---
# (这部分代码保持不变)
try:
    from .darknet import CSPDarknet, BaseConv, CSPLayer, DWConv
    from .yolo import YOLOXHead
except ImportError:
    print("未找到 'nets' 模块... 尝试添加到 sys.path")
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from nets.darknet import CSPDarknet, BaseConv, CSPLayer, DWConv
        from nets.yolo import YOLOXHead
    except ImportError:
        sys.path.append(os.path.abspath(os.path.dirname(__file__)))
        from nets.darknet import CSPDarknet, BaseConv, CSPLayer, DWConv
        from nets.yolo import YOLOXHead


# --- 2. [新模块] YOLOX 的 FPN/PANet ---
# (这部分代码保持不变)
class YOLOX_PANet(nn.Module):
    """
    YOLOX FPN/PANet 颈部.
    """
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"),
                 in_channels=[256, 512, 1024], depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        
        self.in_features = in_features
        # 模块内部缩放: e.g., [256, 512, 1024] * 0.5 -> [128, 256, 512]
        in_channels = [int(width * c) for c in in_channels]

        self.lateral_conv0 = BaseConv(in_channels[2], in_channels[1], 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1]), in_channels[1],
            n=int(3 * depth), shortcut=False, depthwise=depthwise, act=act
        )
        self.reduce_conv1 = BaseConv(in_channels[1], in_channels[0], 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0]), in_channels[0],
            n=int(3 * depth), shortcut=False, depthwise=depthwise, act=act
        )
        self.bu_conv2 = Conv(in_channels[0], in_channels[0], 3, 2, act=act)
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0]), in_channels[1],
            n=int(3 * depth), shortcut=False, depthwise=depthwise, act=act
        )
        self.bu_conv1 = Conv(in_channels[1], in_channels[1], 3, 2, act=act)
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1]), in_channels[2],
            n=int(3 * depth), shortcut=False, depthwise=depthwise, act=act
        )

    def forward(self, backbone_feats):
        x2, x1, x0 = backbone_feats 
        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = F.interpolate(fpn_out0, size=x1.shape[2:4], mode='nearest')
        f_in0 = torch.cat([f_out0, x1], 1)
        f_out1 = self.C3_p4(f_in0) 
        fpn_out1 = self.reduce_conv1(f_out1)
        f_out1 = F.interpolate(fpn_out1, size=x2.shape[2:4], mode='nearest')
        f_in1 = torch.cat([f_out1, x2], 1)
        pan_out2 = self.C3_p3(f_in1) 
        p_in1 = self.bu_conv2(pan_out2)
        p_in1 = torch.cat([p_in1, fpn_out1], 1)
        pan_out1 = self.C3_n3(p_in1) 
        p_in0 = self.bu_conv1(pan_out1)
        p_in0 = torch.cat([p_in0, fpn_out0], 1)
        pan_out0 = self.C3_n4(p_in0) 
        return pan_out2, pan_out1, pan_out0


# --- 3. [你的模块] 时序融合颈 (原封不动) ---
# (这部分代码保持不变)
class Motion_coupling_Neck(nn.Module):
    def __init__(self, in_channels, num_frame=3, act="silu"):
        super().__init__()
        self.num_frame = num_frame
        n_ref = num_frame - 1 
        self.weight = nn.ParameterList(
            nn.Parameter(torch.tensor([1.0 / n_ref]), requires_grad=True) for _ in range(n_ref)
        )
        self.conv_ref = nn.Sequential(
            BaseConv(in_channels * n_ref, in_channels, 1, 1, act=act), 
            BaseConv(in_channels, in_channels, 3, 1, act=act),         
            BaseConv(in_channels, in_channels, 1, 1, act='sigmoid') 
        )
        self.conv_cur = BaseConv(in_channels, in_channels, 3, 1, act=act)
        self.conv_gl = nn.Sequential(
            BaseConv(in_channels*2, in_channels, 1, 1, act=act), 
            BaseConv(in_channels, in_channels, 3, 1, act=act)    
        )
        self.conv_gl_mix = BaseConv(in_channels, in_channels, 3, 1, act=act)
        self.conv_cr_mix = nn.Sequential(
            BaseConv(in_channels*2, in_channels, 1, 1, act=act), 
            BaseConv(in_channels, in_channels, 3, 1, act=act)    
        )
        self.conv_final = nn.Sequential(
            BaseConv(in_channels*2, in_channels, 1, 1, act=act), 
            BaseConv(in_channels, in_channels, 3, 1, act=act)    
        )
    def forward(self, feats):
        key_frame = feats[-1]
        ref_frames = feats[:-1]
        r_feat_cat = torch.cat(ref_frames, dim=1)
        r_feat = self.conv_ref(r_feat_cat)
        c_feat = self.conv_cur(r_feat * key_frame)
        c_feat = self.conv_cr_mix(torch.cat([c_feat, key_frame], dim=1))
        r_feats_list = []
        for i in range(self.num_frame - 1):
            fused_ref_key = torch.cat([ref_frames[i], key_frame], dim=1)
            weighted_feat = self.conv_gl(fused_ref_key) * self.weight[i]
            r_feats_list.append(weighted_feat)
        r_feats_sum = torch.sum(torch.stack(r_feats_list, dim=0), dim=0)
        r_feat_gl = self.conv_gl_mix(r_feats_sum)
        final_fused_feat = self.conv_final(torch.cat([r_feat_gl, c_feat], dim=1))
        return final_fused_feat


# --- 4. [重构+最终修复] 你的 YoloBodySST (计算高效版) ---
class YoloBodySST(nn.Module):
    def __init__(self, num_classes, phi='s', num_frame=3):
        super().__init__()
        self.num_frame = num_frame
        
        depth_dict = {'nano': 0.33, 'tiny': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33}
        width_dict = {'nano': 0.25, 'tiny': 0.375, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        depth, width = depth_dict[phi], width_dict[phi]
        depthwise = True if phi == 'nano' else False
        act = "silu"

        # --- [BUG 修复] ---
        # YOLOX 模块期望的 *基础* 通道数
        base_channels = [256, 512, 1024]
        
        # 我们自己的模块 (Motion_neck) 期望的 *缩放后* 通道数
        # (e.g., for phi='s', width=0.5 -> [128, 256, 512])
        scaled_channels = [int(width * c) for c in base_channels]
        # --- [修复结束] ---
        
        self.backbone_out_features_keys = ("dark3", "dark4", "dark5")
        
        # --- 1. 定义网络组件 ---
        
        # 1a. YOLOX 纯骨干网络 (CSPDarknet)
        try:
            self.backbone = CSPDarknet(
                depth, width, 
                out_features=self.backbone_out_features_keys, 
                depthwise=depthwise, act=act
            )
        except TypeError:
            print("警告: 你的 CSPDarknet 不接受 'out_features' 参数。将使用默认输出。")
            self.backbone = CSPDarknet(
                depth, width, 
                depthwise=depthwise, act=act
            )
        
        # 1b. 你的时序融合颈 (使用 *缩放后* 的通道数)
        self.motion_neck_c3 = Motion_coupling_Neck(in_channels=scaled_channels[0], num_frame=num_frame, act=act)
        self.motion_neck_c4 = Motion_coupling_Neck(in_channels=scaled_channels[1], num_frame=num_frame, act=act)
        self.motion_neck_c5 = Motion_coupling_Neck(in_channels=scaled_channels[2], num_frame=num_frame, act=act)

        # 1c. [保留的优点] 门控残差连接
        self.fusion_gate_c3 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.fusion_gate_c4 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.fusion_gate_c5 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        
        # 1d. YOLOX FPN/PANet 颈部 (使用 *基础* 通道数)
        self.fpn = YOLOX_PANet(
            depth, width, 
            in_features=self.backbone_out_features_keys, 
            in_channels=base_channels, # <-- 修复: 传入 [256, 512, 1024]
            depthwise=depthwise, act=act
        )
        
        # 1e. YOLOX Head (使用 *基础* 通道数)
        self.head = YOLOXHead(
            num_classes, width, 
            in_channels=base_channels, # <-- 修复: 传入 [256, 512, 1024]
            depthwise=depthwise, act=act
        )
        
    def forward(self, inputs):
        # inputs 维度: [bs, num_frame, c, h, w]
        
        c3_list, c4_list, c5_list = [], [], []
        
        # [已修复] 强制按字典键名提取张量
        key_c3, key_c4, key_c5 = self.backbone_out_features_keys
        
        for i in range(self.num_frame):
            backbone_outs = self.backbone(inputs[:, i, :, :, :])
            
            try:
                c3_list.append(backbone_outs[key_c3])
                c4_list.append(backbone_outs[key_c4])
                c5_list.append(backbone_outs[key_c5])
            except (TypeError, KeyError) as e:
                print(f"致命错误: 无法从Backbone的输出中提取特征。")
                print(f"错误: {e}")
                print(f"Backbone的输出类型是: {type(backbone_outs)}")
                if isinstance(backbone_outs, dict):
                    print(f"Backbone输出的键 (Keys) 是: {backbone_outs.keys()}")
                print(f"代码期望的键 (Keys) 是: {self.backbone_out_features_keys}")
                raise
            
        fused_c3 = self.motion_neck_c3(c3_list)
        fused_c4 = self.motion_neck_c4(c4_list)
        fused_c5 = self.motion_neck_c5(c5_list)
        
        final_c3 = c3_list[-1] + self.fusion_gate_c3 * fused_c3
        final_c4 = c4_list[-1] + self.fusion_gate_c4 * fused_c4
        final_c5 = c5_list[-1] + self.fusion_gate_c5 * fused_c5

        fpn_inputs = (final_c3, final_c4, final_c5)
        p3, p4, p5 = self.fpn(fpn_inputs)
        
        fpn_outputs = (p3, p4, p5)
        outputs = self.head(fpn_outputs)
        
        return outputs
# ==================================================== #
# 
#         测试代码 (main 函数) - 与你的原版保持一致
# 
# ==================================================== #
if __name__ == "__main__":
    print("--- 正在运行 YoloBodySST (门控残差版) 模型完整性检查 ---")

    num_classes = 10
    phi = 's'
    num_frame = 3
    
    print(f"正在实例化 YoloBodySST: num_classes={num_classes}, phi='{phi}', num_frame={num_frame}")
    model = YoloBodySST(num_classes=num_classes, phi=phi, num_frame=num_frame)
    model.eval()
    print("模型实例化成功。")
    
    print(f"初始融合门 C3: {model.fusion_gate_c3.item()}")
    print(f"初始融合门 C4: {model.fusion_gate_c4.item()}")
    print(f"初始融合门 C5: {model.fusion_gate_c5.item()}")


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