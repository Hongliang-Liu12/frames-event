import torch
import torch.nn as nn

from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv


# --- 3. 复用您项目中已有的 YOLOXHead ---
from .yolo import YOLOXHead
# --- 1. 引入 SSTNet 的多帧特征提取器 ---
class Feature_Extractor(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), in_channels=[256, 512, 1024], depthwise=False, act="silu"):
        super().__init__()
        # 确保能正确引用您项目中的 darknet 模块
        from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
        self.backbone       = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features    = in_features
        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width), int(in_channels[1] * width), round(3 * depth),
            False, depthwise=depthwise, act=act
        )
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width), int(in_channels[0] * width), round(3 * depth),
            False, depthwise=depthwise, act=act
        )

    def forward(self, input):
        out_features          = self.backbone.forward(input)
        feat1, feat2, feat3   = [out_features[f] for f in self.in_features]
        P5          = self.lateral_conv0(feat3)
        P5_upsample = self.upsample(P5)
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        P5_upsample = self.C3_p4(P5_upsample)
        P4          = self.reduce_conv1(P5_upsample)
        P4_upsample = self.upsample(P4)
        P4_upsample = torch.cat([P4_upsample, feat1], 1)
        P3_out      = self.C3_p3(P4_upsample)
        return P3_out
    

# --- 2. 引入 SSTNet 的时序融合颈 ---
class Motion_coupling_Neck(nn.Module):
    def __init__(self, channels=[128], num_frame=5):
        super().__init__()

        self.num_frame = num_frame
        self.weight = nn.ParameterList(torch.nn.Parameter(torch.tensor([0.25]), requires_grad=True) for _ in range(num_frame))
        self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2, 3, 1),
            BaseConv(channels[0]*2, channels[0], 3, 1, act='sigmoid')
        )
        self.conv_cur = nn.Sequential(BaseConv(channels[0], channels[0], 3, 1), BaseConv(channels[0], channels[0], 3, 1))
        self.conv_gl = nn.Sequential(BaseConv(channels[0]*2, channels[0]*2, 3, 1), BaseConv(channels[0]*2, channels[0], 3, 1))
        self.conv_gl_mix = nn.Sequential(BaseConv(channels[0], channels[0], 3, 1), BaseConv(channels[0], channels[0], 3, 1))
        self.conv_cr_mix = nn.Sequential(BaseConv(channels[0]*2, channels[0]*2, 3, 1), BaseConv(channels[0]*2, channels[0], 3, 1))
        self.conv_final = nn.Sequential(BaseConv(channels[0]*2, channels[0]*2, 3, 1), BaseConv(channels[0]*2, channels[0], 3, 1))

    def forward(self, feats):
        r_feat = torch.cat([feats[j] for j in range(self.num_frame-1)], dim=1)
        r_feat = self.conv_ref(r_feat)
        c_feat = self.conv_cur(r_feat*feats[-1])
        c_feat = self.conv_cr_mix(torch.cat([c_feat, feats[-1]], dim=1))
        r_feats = torch.stack([self.conv_gl(torch.cat([feats[i], feats[-1]], dim=1))*self.weight[i] for i in range(self.num_frame-1)], dim=0)
        r_feat = self.conv_gl_mix(torch.sum(r_feats, dim=0))
        c_feat = self.conv_final(torch.cat([r_feat, c_feat], dim=1))
        return [c_feat] # 返回列表，以适配YOLOXHead
    



# --- 4. 构建新的模型主干: YoloBodySST ---
class YoloBodySST(nn.Module):
    def __init__(self, num_classes, num_frame=3):
        super().__init__()
        # SSTNet 使用的是固定的 depth 和 width 参数
        depth, width = 0.33, 0.50
        self.num_frame = num_frame
        
        # 模块一: 特征提取器 (Backbone)
        # SSTNet的backbone输出通道固定为128，用于后续融合
        self.backbone = Feature_Extractor(depth, width) 

        # 模块二: 时序融合颈 (Neck)
        # 输入通道为128，与backbone输出对应
        self.neck = Motion_coupling_Neck(channels=[128], num_frame=num_frame)
        
        # 模块三: 检测头 (Head)
        # 输入通道也为128，与neck输出对应
        self.head = YOLOXHead(num_classes=num_classes, width=1.0, in_channels=[128], act="silu")
        
    def forward(self, inputs):
        # inputs 的期望维度: [batch, channels, num_frame, height, width]
        
        # 1. 逐帧提取特征
        # feat 将是一个包含 num_frame 个特征图的列表
        feat = [self.backbone(inputs[:, i, :, :, :]) for i in range(self.num_frame)]
        
        # 2. 通过 neck 进行时序特征融合
        feat_fused = self.neck(feat)
        
        # 3. 将融合后的特征送入检测头
        outputs = self.head(feat_fused)
        
        return outputs

# ==================================================== #
#                  新增的测试代码 (main 函数)
# ==================================================== #
if __name__ == "__main__":
    print("--- Running Model Sanity Check ---")



    # 1. 定义模型参数
    num_classes = 10    # 类别数 (例如 VOC a 20)
    num_frame = 4      # 输入的序列长度
    
    # 2. 实例化新模型
    print(f"Instantiating YoloBodySST with num_classes={num_classes}, num_frame={num_frame}...")
    model = YoloBodySST(num_classes=num_classes, num_frame=num_frame)
    model.eval()
    print("Model instantiated successfully.")

    # 3. 创建一个虚拟输入张量
    batch_size = 2
    height, width = 640, 640
    # 输入维度: [batch, channels, num_frame, height, width]
    dummy_input = torch.randn(batch_size, num_frame, 3, height, width)
    print(f"\nCreated a dummy input tensor with shape: {dummy_input.shape}")

    # 4. 执行前向传播
    try:
        print("Performing a forward pass...")
        with torch.no_grad():
            outputs = model(dummy_input)
        print("Forward pass completed successfully.")

        # 5. 打印输出维度
        print("\n--- Output Shapes ---")
        if isinstance(outputs, (list, tuple)):
            for i, out in enumerate(outputs):
                print(f"Output [{i}] shape: {out.shape}")
        else:
            print(f"Output shape: {outputs.shape}")
        
        # 验证输出
        # YOLOXHead 对于 in_channels=[128] 的情况，只会有一个输出
        expected_channels = num_classes + 5
        # backbone 的输出是 (B, 128, H/8, W/8)，所以 head 的输出也是这个尺寸
        expected_h, expected_w = height // 8, width // 8
        
        output_shape = outputs[0].shape
        print(f"\nExpected output shape: [{batch_size}, {expected_channels}, {expected_h}, {expected_w}]")

        if len(outputs) == 1 and list(output_shape) == [batch_size, expected_channels, expected_h, expected_w]:
             print("\n[SUCCESS] The model output shape is correct!")
        else:
             print("\n[FAILURE] The model output shape is NOT correct.")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during the forward pass: {e}")
        import traceback
        traceback.print_exc()