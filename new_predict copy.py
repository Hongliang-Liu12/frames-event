from nets.yolo import YoloBody

import torch
import cv2
import numpy as np
import os
import torchvision
from loguru import logger

def postprocess(prediction, num_classes=10, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output 

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    _COLORS = np.array(
        [
            [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933],
            [0.635, 0.078, 0.184], [0.300, 0.300, 0.300], [0.600, 0.600, 0.600],
            [1.000, 0.000, 0.000], [1.000, 0.500, 0.000], [0.749, 0.749, 0.000],
            [0.000, 1.000, 0.000], [0.000, 0.000, 1.000], [0.667, 0.000, 1.000],
        ]
    )
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img
cls_names=(
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",

)
def visual(output, img_info, cls_conf=0.35):
    ratio = img_info["ratio"]
    img = img_info["raw_img"]
    if output is None:
        return img
    output = output.cpu()
    bboxes = output[:, 0:4]
    bboxes /= ratio
    cls = output[:, 6]
    scores = output[:, 4] * output[:, 5]
    vis_res = vis(img, bboxes, scores, cls, cls_conf, cls_names)
    return vis_res

if __name__ == "__main__":

    #===============加载模型权重================#
    # 类别数
    num_classes = 10 
    # 模型类型，这里以's'为例
    phi = 's'
    # 创建YoloBody模型
    model = YoloBody(num_classes=num_classes, phi=phi)
    print(model.backbone)
    # 打印正在加载的权重文件路径
    model_path ='/home/lhl/Git/yolox-pytorch-bilibili/yolox_only_weights.pth'
    # model_path ='/home/lhl/Git/yolox-bilibili-event/logs/ep097-loss2.092-val_loss4.297.pth'
    print(f"正在从 {model_path} 加载模型权重...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)

    #使用 model.load_state_dict() 将参数加载进模型
    # strict=False 可以在模型和权重文件结构不完全匹配时，只加载匹配的部分
    model.load_state_dict(state_dict, strict=True)
    print("模型权重加载完成！")

    print('Weights successfully loaded into the model.')
    model.eval()

    #================读取测试照片,进行letterbox变换================#
    # 创建一个虚拟输入张量 (Batch, Channels, Height, Width)
    path = "/home/lhl/Git/yolox-bilibili-event/test.png"

    img_info = {"id": 0}
    img_info["file_name"] = os.path.basename(path)
    img = cv2.imread(path)
    # print("原始图片 - 尺寸:", img.shape, "类型:", img.dtype, "值:", img[:, 0, 0])
    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = img
    # --- 直观的图片预处理代码 (Letterbox) ---
    # 定义模型需要的输入尺寸，这里假设是 640x640
    input_size = (640, 640)
        # 获取原始图片的尺寸
    img_h, img_w = img.shape[:2]
    # 计算缩放比例，以保持宽高比
    scale = min(input_size[0] / img_h, input_size[1] / img_w)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    img_info["ratio"] = scale
    # 缩放图片
    # img=img.transpose((2, 0, 1))  # HWC to CHW
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    # 创建一个空的灰色背景画布 (114是灰度值)
    # 注意：cv2的颜色顺序是BGR，所以需要(114, 114, 114)
    padded_img = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    # 将缩放后的图片粘贴到灰色画布的左上角
    # [height_start:height_end, width_start:width_end]
    padded_img[0:new_h, 0:new_w] = resized_img
    padded_img = padded_img.transpose((2,0,1))
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    # 打印处理后的图片尺寸，确认是 640x640
    print(f"原始图片尺寸: {img_w}x{img_h}")
    print(f"处理后图片尺寸: {padded_img.shape[1]}x{padded_img.shape[0]}")

    # print("preproc处理后 - 尺寸:", padded_img.shape, "类型:", padded_img.dtype, "值:", padded_img[0, 0, :])
    # # --- 保存和显示处理后的图片 ---
    # # 定义保存路径
    # save_folder = "./inference_results_preprocessed"
    # os.makedirs(save_folder, exist_ok=True)
    # save_path = os.path.join(save_folder, "preprocessed_" + os.path.basename(path))

    # # 保存图片
    # cv2.imwrite(save_path, padded_img)
    # print(f"处理后的图片已保存到: {save_path}")
    # print("preproc处理后 - 尺寸:", padded_img.shape, "类型:", padded_img.dtype, "值:", padded_img[0, 0, :])
    #torch张量，并添加batch维度
    img = torch.from_numpy(padded_img).unsqueeze(0)
    img = img.float()




    #============获取FPN骨干网络的输出============#
    # 获取FPN的输出特征图
    fpn_outs = model.backbone(img)
    # print("--- FPN 输出维度 ---")
    # print(f"P3_out: {fpn_outs[0].shape}")
    # print(fpn_outs[0][0,0,0,:])
    # print(f"P4_out: {fpn_outs[1].shape}")
    # print(fpn_outs[1][0,0,0,:])
    # print(f"P5_out: {fpn_outs[2].shape}")
    # print(fpn_outs[2][0,0,0,:])



    # 获取YOLOXHead的输出
    head_outputs = model.head(fpn_outs)
    decode_in_inference=True

    # 1. 获取每个特征层的高和宽
    hw = [x.shape[-2:] for x in head_outputs]
    # 2. 定义 stride 的值
    strides = [8, 16, 32]
    # [batch, n_anchors_all, 85]
    outputs = torch.cat(
        [x.flatten(start_dim=2) for x in head_outputs], dim=2
    ).permute(0, 2, 1)

    # print("outputs")
    # print(outputs.shape)
    # print(outputs[0,0,:  ])

    grids = []
    strides_tensor_list = []
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
    for (hsize, wsize), stride in zip(hw, strides):
        yv, xv =  torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing="ij")
        grid = torch.stack((xv, yv), 2).view(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        strides_tensor_list.append(torch.full((*shape, 1), stride))

    grids = torch.cat(grids, dim=1).type(outputs.dtype)
    strides_tensor = torch.cat(strides_tensor_list, dim=1).type(outputs.dtype)

    decoded_outputs = torch.cat([
        (outputs[..., 0:2] + grids) * strides_tensor,
        torch.exp(outputs[..., 2:4]) * strides_tensor,
        outputs[..., 4:]
    ], dim=-1)

    # print("decoded_outputs")
    # print(decoded_outputs.shape)
    # print(decoded_outputs[0,0,:  ])
    #这里获取了模型的整体输出 torch.Size([1, 8400, 15])


    #进行postprocess，即非极大值抑制
    outputs = postprocess(decoded_outputs, num_classes=10, conf_thre=0.3, nms_thre=0.3, class_agnostic=True)
    # print("postprocess 后的 outputs")
    # print(len(outputs))
    # print(outputs[0].shape)
    # print(outputs[0][0,:  ])
    
    # print(outputs,img_info)


    vis_folder = "./inference_results"
    os.makedirs(vis_folder, exist_ok=True)
    if outputs[0] is not None:
        result_image = visual(outputs[0], img_info, cls_conf=0.3)
        save_file_name = os.path.join(vis_folder, f"detected_{os.path.basename(path)}")
        logger.info(f"Saving detection result in {save_file_name}")
        cv2.imwrite(save_file_name, result_image)


    # print("\n--- YOLOXHead 输出维度 ---")
    # print(f"P3_head_output: {head_outputs[0].shape}")
    # print(head_outputs[0][0,:])
    # print(f"P4_head_output: {head_outputs[1].shape}")
    # print(head_outputs[1][0,:])
    # print(f"P5_head_output: {head_outputs[2].shape}")
    # print(head_outputs[2][0,:])

    # 获取整个模型的输出
    final_outputs= model(img)
    # print("\n--- 整个模型输出维度 ---")
    # print(f"P3_final_output: {final_outputs[0].shape}")
    # print(f"P4_final_output: {final_outputs[1].shape}")
    # print(f"P5_final_output: {final_outputs[2].shape}")