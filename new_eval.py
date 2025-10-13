import os
import torch
import numpy as np
import cv2
import torchvision

from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from PIL import Image

from nets.yolo import YoloBody
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# --- Postprocess function from predict_test.py ---
def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
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

# --- Dataset Definition ---
CLASS_NAMES = [
    "people", "car", "bicycle", "electric bicycle",
    "basketball", "ping_pong", "goose", "cat", "bird", "UAV"
]

class Evdet200kCocoDataset(CocoDetection):
    def __init__(self, root_dir, split="test"):
        annotation_path = os.path.join(root_dir, "Event_Frame", "annotations", f"{split}.json")
        images_root = os.path.join(root_dir, "Event_Frame", "data")
        super().__init__(root=images_root, annFile=annotation_path)

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.root, img_info['file_name'])
        image_np = cv2.imread(image_path)
        # BGR to RGB is handled inside the collate_fn now if needed, but cv2 reads as BGR
        return image_np, target, img_info

# --- CORRECTED Collate Function ---
def letterbox_collate_fn_corrected(batch):
    images, targets, img_infos = zip(*batch)
    
    processed_images = []
    ratios = []
    
    input_size = (640, 640) # Model input size

    for img in images:
        # This block is a direct copy of the logic from predict_test.py
        img_h, img_w = img.shape[:2]
        scale = min(input_size[0] / img_h, input_size[1] / img_w)
        ratios.append(scale)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Manually create the padded image
        padded_img = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
        padded_img[0:new_h, 0:new_w] = resized_img
        
        # Transpose and make contiguous
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        
        processed_images.append(padded_img)
        
    # Stack and convert to tensor
    images_batch = torch.from_numpy(np.array(processed_images))
    
    return images_batch, list(targets), list(img_infos), ratios

# --- NEW FUNCTION: Print per-class results ---
def print_per_class_results(coco_eval, class_names):
    """打印每个类别的 AP@[0.5:0.95] 和 AR@100 结果。"""
    eval_results = coco_eval.eval
    precisions = eval_results['precision'] # [T, R, K, A, M]
    recalls = eval_results['recall']     # [T, K, A, M]
    
    num_classes = len(class_names)
    ap_per_class = []
    ar_per_class = []
    
    # T=0:10 (IoU=0.5:0.95), R=0:101 (Recall points), A=0 (Area=all), M=2 (maxDets=100)

    for k in range(num_classes):
        # 1. 计算 AP@[0.5:0.95]
        # 对所有 IoU 和所有 Recall 阈值求平均
        p = precisions[:, :, k, 0, 2]
        # 排除 NaN/无效值 (-1)
        p = p[p > -1] 
        ap = np.mean(p) * 100.0 if p.size > 0 else np.nan
        ap_per_class.append((class_names[k], ap))

        # 2. 计算 AR@100
        # 对所有 IoU 阈值求平均 (recall 数组已经是最大召回率)
        r = recalls[:, k, 0, 2]
        # 排除 NaN/无效值 (-1)
        r = r[r > -1]
        ar = np.mean(r) * 100.0 if r.size > 0 else np.nan
        ar_per_class.append((class_names[k], ar))
        
    def format_table(title, data, metric_name):
        print(f"\n{title}:")
        num_cols = 3
        CLASS_W = 18 # 最长类名 "electric bicycle" (16字符) + 缓冲
        METRIC_W = 7 # "XX.XXX" 7字符
        
        # 1. 打印头部
        header_cols = []
        for _ in range(num_cols):
             # 左对齐 'class'，右对齐 'AP' 或 'AR'
             header_cols.append(f"{'class':<{CLASS_W}} {metric_name:>{METRIC_W}}")
        # 使用 3 个空格分隔列
        print("   ".join(header_cols)) 
        
        # 2. 打印分隔线 (可选，但有助于视觉分隔) 
        
        # # 3. 打印数据行 # 按列主序索引，即先填满第一列再填第二列
        # num_rows = (len(data) + num_cols - 1) // num_cols
        
        # for r in range(num_rows):
        #     row_parts = []
        #     for c in range(num_cols):
        #         idx = r + c * num_rows 
        #         if idx < len(data):
        #             class_name, metric_value = data[idx]
        #             # 格式化数值为三位小数
        #             metric_str = f"{metric_value:.3f}" if not np.isnan(metric_value) else "-"
                    
        #             # 使用 ljust() 左对齐类名，使用 rjust() 右对齐数值
        #             row_parts.append(f"{class_name.ljust(CLASS_W)} {metric_str.rjust(METRIC_W)}")
        #         else:
        #             # 填充空单元格
        #             row_parts.append(f"{' '.ljust(CLASS_W)} {' '.ljust(METRIC_W)}")
        #     print("   ".join(row_parts)) # 使用 3 个空格分隔列
            # 3. 打印数据行 (修改后的部分)
        # 按行主序进行遍历和打印
        for i in range(0, len(data), num_cols):
            # 从数据列表中一次性取出当前行的所有元素
            row_items = data[i:i + num_cols]
            
            row_parts = []
            for class_name, metric_value in row_items:
                # 格式化每个单元格
                metric_str = f"{metric_value:.3f}" if not np.isnan(metric_value) else "-"
                row_parts.append(f"{class_name.ljust(CLASS_W)} {metric_str.rjust(METRIC_W)}")
                
            # 如果当前行不满，则用空格补齐，确保对齐
            while len(row_parts) < num_cols:
                row_parts.append(f"{' '.ljust(CLASS_W)} {' '.ljust(METRIC_W)}")
                
            print("   ".join(row_parts))
            
    format_table("per class AP", ap_per_class, "AP")
    format_table("per class AR", ar_per_class, "AR")
    print("="*50)


# --- Main Evaluation Logic ---
def evaluate_model():
    root_dir = "/home/lhl/Git/datasets/EvDET200K"
    model_path = "/home/lhl/Git/yolox-bilibili-event/logs/ep017-loss3.500-val_loss4.229.pth"
    phi = 's'
    num_classes = len(CLASS_NAMES)
    input_shape = (640, 640)
    confidence = 0.01
    nms_iou = 0.65
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading dataset...")
    val_dataset = Evdet200kCocoDataset(root_dir, split="test")
    # Use the corrected collate function
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True, collate_fn=letterbox_collate_fn_corrected)
    
    coco_gt = val_dataset.coco
    
    # print(f"Loading model from {model_path}...")
    # model = YoloBody(num_classes=num_classes, phi=phi)
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model = model.to(device).eval()
    # print("Model loaded.")

    print(f"Loading model from {model_path}...")

    # 1. 加载整个检查点文件
    # 这一步会加载一个包含各种信息的字典
    checkpoint = torch.load(model_path, map_location=device)

    # 2. 提取模型的 state_dict
    # YOLOX 官方和训练检查点通常将模型的 state_dict 存储在 'model' 键下。
    # 否则，假设加载的内容已经是纯净的 state_dict。
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_state_dict = checkpoint['model']
        print("Checkpoint is a full training checkpoint. Extracted 'model' state_dict.")
    else:
        # 适用于只保存了纯净权重文件的情况 (例如: torch.save(model.state_dict(), ...))
        model_state_dict = checkpoint
        print("Checkpoint is a pure state_dict file.")

    # 3. 创建模型并加载提取出的权重
    model = YoloBody(num_classes=num_classes, phi=phi)
    model.load_state_dict(model_state_dict)

    # 4. 移动到设备并设置评估模式
    model = model.to(device).eval()
    print("Model loaded successfully with weights only.")

    print("Starting evaluation...")
    results = []
    
    strides = [8, 16, 32]
    hw = [(int(input_shape[0] / s), int(input_shape[1] / s)) for s in strides]
    
    for images, targets, img_infos, ratios in tqdm(val_dataloader, desc="Evaluating"):
        images = images.to(device).float() # Ensure tensor is float
        
        with torch.no_grad():
            outputs = model(images)
            
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])

            grids = []
            strides_tensor_list = []
            for (hsize, wsize), stride in zip(hw, strides):
                yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing="ij")
                grid = torch.stack((xv, yv), 2).view(1, -1, 2)
                grids.append(grid)
                shape = grid.shape[:2]
                strides_tensor_list.append(torch.full((*shape, 1), stride))

            grids = torch.cat(grids, dim=1).type(outputs.dtype).to(device)
            strides_tensor = torch.cat(strides_tensor_list, dim=1).type(outputs.dtype).to(device)

            decoded_outputs = torch.cat([
                (outputs[..., 0:2] + grids) * strides_tensor,
                torch.exp(outputs[..., 2:4]) * strides_tensor,
                outputs[..., 4:]
            ], dim=-1)

            final_outputs = postprocess(decoded_outputs, num_classes, confidence, nms_iou, class_agnostic=False)


            for batch_idx, output_per_image in enumerate(final_outputs):
                if output_per_image is not None:
                    final_outputs_cpu = output_per_image.cpu().numpy()
                    top_label = final_outputs_cpu[:, 6].astype('int32')
                    top_conf = final_outputs_cpu[:, 4] * final_outputs_cpu[:, 5]
                    top_boxes = final_outputs_cpu[:, :4]

                    # 关键：使用每个图片对应的ratio和img_info，而不是只用第0个
                    ratio = ratios[batch_idx]
                    image_shape = img_infos[batch_idx] 
                    h, w = image_shape['height'], image_shape['width']

                    top_boxes /= ratio
                    top_boxes[:, [0, 2]] = top_boxes[:, [0, 2]].clip(0, w)
                    top_boxes[:, [1, 3]] = top_boxes[:, [1, 3]].clip(0, h)

                    for i, c in enumerate(top_label):
                        predicted_class = c + 1
                        box = top_boxes[i]
                        score = float(top_conf[i])
                        x1, y1, x2, y2 = box
                        
                        coco_box = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                        
                        result = {
                            "image_id": img_infos[batch_idx]['id'], # 关键：使用正确的image_id
                            "category_id": predicted_class,
                            "bbox": coco_box,
                            "score": score,
                        }
                        results.append(result)


    if not results:
        print("No detections were made after processing all images.")
        return
        
    print("\nEvaluating COCO mAP...")
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    
    # 这一步计算所有指标并将结果存储在 coco_eval.eval 中
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # 打印标准 COCO 总结指标
    coco_eval.summarize()
    
    # --- 新增：打印每个类别的 AP/AR ---
    print_per_class_results(coco_eval, CLASS_NAMES)

if __name__ == "__main__":
    evaluate_model()