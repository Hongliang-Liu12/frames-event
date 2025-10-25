import torch
import torch.nn as nn
import sys
import os

# --- 1. 从您原有的 "frames" 项目中导入核心模块 ---
try:
    from .yolo import YOLOPAFPN, YOLOXHead 
    from .darknet import BaseConv
except ImportError:
    # 修复作为独立脚本运行时可能出现的导入错误
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nets.yolo import YOLOPAFPN, YOLOXHead
    from nets.darknet import BaseConv


# --- 2. 引入并修正 SSTNet 的时序融合颈 (Motion_coupling_Neck) ---
class Motion_coupling_Neck(nn.Module):
    # --- 函数签名保持不变 ---
    def __init__(self, in_channels, num_frame=3, act="silu"):
        super().__init__()
        self.num_frame = num_frame
        n_ref = num_frame - 1 # 参考帧数量
        
        # 权重参数 (保持不变)
        self.weight = nn.ParameterList(
            nn.Parameter(torch.tensor([1.0 / n_ref]), requires_grad=True) for _ in range(n_ref)
        )

        self.conv_ref = nn.Sequential(
            BaseConv(in_channels * n_ref, in_channels, 1, 1, act=act), # 1x1 瓶颈层
            BaseConv(in_channels, in_channels, 3, 1, act=act),       # 浅层 3x3
            BaseConv(in_channels, in_channels, 1, 1, act='sigmoid') # 1x1 门
        )
        
        self.conv_cur = BaseConv(in_channels, in_channels, 3, 1, act=act)

        self.conv_gl = nn.Sequential(
            BaseConv(in_channels*2, in_channels, 1, 1, act=act), # 1x1 瓶颈层
            BaseConv(in_channels, in_channels, 3, 1, act=act)    # 3x3 卷积
        )

        self.conv_gl_mix = BaseConv(in_channels, in_channels, 3, 1, act=act)

        self.conv_cr_mix = nn.Sequential(
            BaseConv(in_channels*2, in_channels, 1, 1, act=act), # 1x1 瓶颈层
            BaseConv(in_channels, in_channels, 3, 1, act=act)    # 3x3 卷积
        )
        
        self.conv_final = nn.Sequential(
            BaseConv(in_channels*2, in_channels, 1, 1, act=act), # 1x1 瓶颈层
            BaseConv(in_channels, in_channels, 3, 1, act=act)    # 3x3 卷积
        )
    def forward(self, feats):
        key_frame = feats[-1]
        ref_frames = feats[:-1]

        # 分支1: 融合所有参考帧，再作用于关键帧
        r_feat_cat = torch.cat(ref_frames, dim=1)
        r_feat = self.conv_ref(r_feat_cat)
        c_feat = self.conv_cur(r_feat * key_frame)
        c_feat = self.conv_cr_mix(torch.cat([c_feat, key_frame], dim=1))
        
        # 分支2: 将每个参考帧分别与关键帧融合，然后加权求和
        r_feats_list = []
        for i in range(self.num_frame - 1):
            fused_ref_key = torch.cat([ref_frames[i], key_frame], dim=1)
            weighted_feat = self.conv_gl(fused_ref_key) * self.weight[i]
            r_feats_list.append(weighted_feat)
        
        r_feats_sum = torch.sum(torch.stack(r_feats_list, dim=0), dim=0)
        r_feat_gl = self.conv_gl_mix(r_feats_sum)

        # 最终融合两个分支
        final_fused_feat = self.conv_final(torch.cat([r_feat_gl, c_feat], dim=1))
        
        return final_fused_feat

# --- 3. 构建新的“多尺度-多帧”模型主干 (采用门控残差连接) ---
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

        self.neck_p3 = Motion_coupling_Neck(in_channels=p3_channels, num_frame=num_frame, act=act)
        self.neck_p4 = Motion_coupling_Neck(in_channels=p4_channels, num_frame=num_frame, act=act)
        self.neck_p5 = Motion_coupling_Neck(in_channels=p5_channels, num_frame=num_frame, act=act)

        # --- 核心修改 1: 引入可学习的“融合门”，并初始化为0 ---
        self.fusion_gate_p3 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.fusion_gate_p4 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.fusion_gate_p5 = nn.Parameter(torch.tensor([0.0]), requires_grad=True)

        self.head = YOLOXHead(num_classes, width, in_channels=in_channels_config, depthwise=depthwise, act=act)
        
    def forward(self, inputs):
        # inputs 维度: [bs, num_frame, c, h, w]
        
        p3_features, p4_features, p5_features = [], [], []
        for i in range(self.num_frame):
            p3, p4, p5 = self.backbone(inputs[:, i, :, :, :])
            p3_features.append(p3)
            p4_features.append(p4)
            p5_features.append(p5)
            
        fused_p3 = self.neck_p3(p3_features)
        fused_p4 = self.neck_p4(p4_features)
        fused_p5 = self.neck_p5(p5_features)
        
        original_p3 = p3_features[-1]
        original_p4 = p4_features[-1]
        original_p5 = p5_features[-1]

        # --- 核心修改 2: 使用“门”来控制时序信息的融合 ---
        # 训练初期，gate为0，模型等价于原始YOLOX
        # 训练过程中，网络会自主学习gate的大小
        final_p3 = original_p3 + self.fusion_gate_p3 * fused_p3
        final_p4 = original_p4 + self.fusion_gate_p4 * fused_p4
        final_p5 = original_p5 + self.fusion_gate_p5 * fused_p5

        # final_p3 = original_p3
        # final_p4 = original_p4 
        # final_p5 = original_p5 

        # final_p3 = original_p3 + self.fusion_gate_p3 * fused_p3
        # final_p4 = original_p4 
        # final_p5 = original_p5 


        final_features = (final_p3, final_p4, final_p5)
        # final_features = (original_p3, original_p4, original_p5)  # 暂时注释掉融合，测试原始YOLOX功能正常
        outputs = self.head(final_features)
        
        return outputs

# ==================================================== #
#                   测试代码 (main 函数)
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
    # 打印新增的gate参数
    print(f"初始融合门 P3: {model.fusion_gate_p3.item()}")
    print(f"初始融合门 P4: {model.fusion_gate_p4.item()}")
    print(f"初始融合门 P5: {model.fusion_gate_p5.item()}")


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

