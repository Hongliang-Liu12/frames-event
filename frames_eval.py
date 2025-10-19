import os
import torch
import numpy as np
import cv2
import torchvision
import json
from nets.yolo_training import YOLOLoss, weights_init
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from tqdm import tqdm
# ↓↓↓ 你需要从这里导入你的 *新* 模型 ↓↓↓
# 假设你的新模型仍然叫 YoloBody 并且在 nets.yolo 中
from nets.yolo_frames_net import YoloBodySST
# from nets.yolo_seq import YoloBody # 或者你新模型的实际路径
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    # ... (此函数保持不变)
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
            nms_out_index = torchvision.ops.nms(detections[:, :4], detections[:, 4] * detections[:, 5], nms_thre)
        else:
            nms_out_index = torchvision.ops.batched_nms(detections[:, :4], detections[:, 4] * detections[:, 5], detections[:, 6], nms_thre)

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    return output

CLASS_NAMES = [
    "people", "car", "bicycle", "electric bicycle",
    "basketball", "ping_pong", "goose", "cat", "bird", "UAV"
]

# ================================================================ #
#                  ↓↓↓ MODIFIED DATASET ↓↓↓
# ================================================================ #
class Evdet200kCocoDataset(CocoDetection):
    def __init__(self, root_dir, split="test", seq_len=3): # <--- MODIFIED: 添加 seq_len
        annotation_path = os.path.join(root_dir, "Event_Frame", "annotations", f"{split}.json")
        images_root = os.path.join(root_dir, "Event_Frame", "data")
        super().__init__(root=images_root, annFile=annotation_path)
        
        self.seq_len = seq_len # <--- MODIFIED
        
        # --- MODIFIED: 借用新 EventJsonDataset 中的序列构建逻辑 ---
        print(f"Building sequences for {split} set (seq_len={self.seq_len})...")
        groups = {}
        # 1. 按文件夹分组
        for img_id in self.ids:
            img_info = self.coco.loadImgs(img_id)[0]
            file_name = img_info.get('file_name', '')
            folder = os.path.dirname(file_name)
            groups.setdefault(folder, []).append((img_id, file_name))

        id_to_sequence = {}
        # 2. 对每个组排序并创建滑动窗口
        for folder, items in groups.items():
            try:
                # 按文件名中的数字排序
                items_sorted = sorted(items, key=lambda x: int(os.path.splitext(os.path.basename(x[1]))[0]))
            except Exception:
                # 如果失败，则按字典序排序
                items_sorted = sorted(items, key=lambda x: x[1])
            
            ids_sorted = [it[0] for it in items_sorted]
            
            # 3. 创建映射：target_id -> [id_t-N+1, ..., id_t]
            for i in range(self.seq_len - 1, len(ids_sorted)):
                target_id = ids_sorted[i] # 目标帧是窗口的最后一帧
                seq_ids = ids_sorted[i - self.seq_len + 1 : i + 1]
                id_to_sequence[target_id] = seq_ids

        self.id_to_sequence = id_to_sequence
        
        # 4. 过滤 self.ids，只保留那些作为有效序列结尾的图像ID
        original_id_count = len(self.ids)
        self.ids = [img_id for img_id in self.ids if img_id in self.id_to_sequence]
        print(f"Filtered {original_id_count} -> {len(self.ids)} valid target frames.")
        # --- END MODIFIED ---

    def __getitem__(self, index):
        # <--- MODIFIED: 加载整个序列，但目标只针对最后一帧 ---
        
        # 1. 获取目标帧 (最后一帧) 的 ID 和信息
        img_id = self.ids[index] # 这是目标帧 (t) 的 ID
        target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
        target_img_info = self.coco.loadImgs(img_id)[0]
        
        # 2. 获取这个目标帧对应的完整序列 ID [t-N+1, ..., t]
        seq_ids = self.id_to_sequence[img_id]
        
        image_seq_np = []
        
        # 3. 加载序列中的每一帧
        for frame_id in seq_ids:
            frame_info = self.coco.loadImgs(frame_id)[0]
            image_path = os.path.join(self.root, frame_info['file_name'])
            image_np = cv2.imread(image_path)
            
            # 处理可能丢失的图像
            if image_np is None:
                print(f"Warning: Could not read image {image_path}. Using black frame.")
                h = target_img_info.get('height', 640)
                w = target_img_info.get('width', 640)
                image_np = np.zeros((h, w, 3), dtype=np.uint8)
                
            image_seq_np.append(image_np)

        # 返回：
        # 1. 图像序列 (list of np.array)
        # 2. 目标帧的标注
        # 3. 目标帧的图像信息
        return image_seq_np, target, target_img_info
        # --- END MODIFIED ---

# ================================================================ #
#                  ↓↓↓ MODIFIED COLLATE_FN ↓↓↓
# ================================================================ #
def letterbox_collate_fn(batch):
    # batch 是一个列表, 每一项是 (image_seq_np, target, target_img_info)
    images_seqs, targets, img_infos = zip(*batch)
    
    processed_seqs = []
    ratios = [] # <--- MODIFIED: 我们只关心目标帧(最后一帧)的ratio
    input_size = (640, 640) # 假设输入尺寸
    
    # 遍历批次中的每个序列
    for seq in images_seqs:
        processed_frames = []
        
        # 遍历序列中的每一帧
        for i, img in enumerate(seq):
            img_h, img_w = img.shape[:2]
            scale = min(input_size[0] / img_h, input_size[1] / img_w)
            
            # <--- MODIFIED: 只保存最后一帧 (目标帧) 的缩放比例
            if i == len(seq) - 1:
                ratios.append(scale)
                
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            padded_img = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
            padded_img[0:new_h, 0:new_w] = resized_img
            
            padded_img = padded_img.transpose((2, 0, 1)) # HWC -> CHW
            padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
            processed_frames.append(padded_img)
            
        # 将处理后的帧堆叠成 (N, C, H, W)
        processed_seqs.append(np.stack(processed_frames, axis=0))

    # 将所有序列堆叠成 (BS, N, C, H, W)
    images_batch = torch.from_numpy(np.stack(processed_seqs, axis=0))
    
    # 返回 (BS, N, C, H, W) 的图像，以及 *只对应目标帧* 的 targets, img_infos, ratios
    return images_batch, list(targets), list(img_infos), ratios
    # --- END MODIFIED ---

def print_per_class_results(coco_eval, class_names):
    # ... (此函数保持不变)
    eval_results = coco_eval.eval
    precisions = eval_results['precision']
    recalls = eval_results['recall']
    cat_ids = coco_eval.cocoGt.getCatIds()
    id_to_name = {cat['id']: cat['name'] for cat in coco_eval.cocoGt.loadCats(cat_ids)}
    
    results_data = []
    for k_idx, cat_id in enumerate(cat_ids):
        p = precisions[:, :, k_idx, 0, 2]
        p = p[p > -1]
        ap = np.mean(p) * 100.0 if p.size > 0 else float('nan')
        r = recalls[:, k_idx, 0, 2]
        r = r[r > -1]
        ar = np.mean(r) * 100.0 if r.size > 0 else float('nan')
        results_data.append((id_to_name.get(cat_id, "unknown"), ap, ar))

    print("\n" + "="*70)
    print(f"{'CLASS':<20} | {'AP @[.5:.95]':^20} | {'AR @[100]':^20}")
    print("-" * 70)
    for name, ap, ar in results_data:
        ap_str = f"{ap:.3f}" if not np.isnan(ap) else "---"
        ar_str = f"{ar:.3f}" if not np.isnan(ar) else "---"
        print(f"{name:<20} | {ap_str:^20} | {ar_str:^20}")
    print("=" * 70)


# ================================================================ #
#                  ↓↓↓ EVALUATION FUNCTION (UNMODIFIED) ↓↓↓
# ================================================================ #
def get_coco_map(model, dataloader, coco_gt, device, confidence=0.01, nms_iou=0.65):
    """
    此函数 *无需修改*。
    
    - 它接收的 'images' 已经是 (BS, N, C, H, W) 形状，并将其送入模型。
    - 它接收的 'img_infos' 和 'ratios' 已经与目标帧(最后一帧)对应。
    - 模型的 'outputs' 格式不变。
    - 后续的解码、NMS 和坐标反算逻辑因此保持完全兼容。
    """
    model.eval()
    
    num_classes = len(CLASS_NAMES)
    input_shape = (640, 640)
    results = []
    strides = [8,16,32]
    hw = [(int(input_shape[0] / s), int(input_shape[1] / s)) for s in strides]
    
    print("Starting evaluation...")
    # <--- 注意: 这里的 'images' 将是 (BS, N, C, H, W) 形状
    for images, _, img_infos, ratios in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images) # <--- 模型现在接收 (BS, N, C, H, W)
            
            # --- 后面的所有逻辑保持不变 ---
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
            outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
            grids, strides_tensor_list = [], []
            for (hsize, wsize), stride in zip(hw, strides):
                yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)], indexing="ij")
                grid = torch.stack((xv, yv), 2).view(1, -1, 2)
                grids.append(grid)
                strides_tensor_list.append(torch.full((*grid.shape[:2], 1), stride))
            grids = torch.cat(grids, dim=1).type(outputs.dtype).to(device)
            strides_tensor = torch.cat(strides_tensor_list, dim=1).type(outputs.dtype).to(device)
            decoded_outputs = torch.cat([(outputs[..., 0:2] + grids) * strides_tensor, torch.exp(outputs[..., 2:4]) * strides_tensor, outputs[..., 4:]], dim=-1)
            final_outputs = postprocess(decoded_outputs, num_classes, confidence, nms_iou)

        # 格式化结果 (完全不变)
        for batch_idx, output_per_image in enumerate(final_outputs):
            if output_per_image is not None:
                final_outputs_cpu = output_per_image.cpu().numpy()
                top_label = final_outputs_cpu[:, 6].astype('int32')
                top_conf = final_outputs_cpu[:, 4] * final_outputs_cpu[:, 5]
                top_boxes = final_outputs_cpu[:, :4]
                
                # 'ratio' 和 'img_info' 已经正确对应到这一帧
                ratio = ratios[batch_idx]
                h, w = img_infos[batch_idx]['height'], img_infos[batch_idx]['width']
                
                top_boxes /= ratio
                top_boxes[:, [0, 2]] = np.clip(top_boxes[:, [0, 2]], 0, w)
                top_boxes[:, [1, 3]] = np.clip(top_boxes[:, [1, 3]], 0, h)
                
                for i, c in enumerate(top_label):
                    predicted_class_id = coco_gt.getCatIds(catNms=[CLASS_NAMES[c]])[0]
                    box, score = top_boxes[i], float(top_conf[i])
                    x1, y1, x2, y2 = box
                    coco_bbox = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                    results.append({
                        "image_id": img_infos[batch_idx]['id'],
                        "category_id": predicted_class_id,
                        "bbox": coco_bbox,
                        "score": score
                    })

    if not results:
        print("No detections were made. Cannot evaluate.")
        return None

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    return coco_eval

# ================================================================ #
#                  ↓↓↓ STANDALONE EXECUTION EXAMPLE ↓↓↓
# ================================================================ #
if __name__ == "__main__":
    # --- 配置 ---
    DATASET_ROOT_DIR = "/home/lhl/Git/datasets/EvDET200K"
    MODEL_PATH = "/home/lhl/Git/frames-event/log_frames/ep003-loss7.086.pth"
    BATCH_SIZE = 4
    CONFIDENCE = 0.01
    NMS_IOU = 0.65
    CUDA = True
    SEQ_LEN = 3 # <--- MODIFIED: 定义序列长度

    # --- 1. Setup device and dataset ---
    device = torch.device('cuda' if CUDA and torch.cuda.is_available() else 'cpu')
    print("Loading dataset...")
    # <--- MODIFIED: 使用修改后的 Dataset，并传入 seq_len
    val_dataset = Evdet200kCocoDataset(DATASET_ROOT_DIR, split="test", seq_len=SEQ_LEN)
    
    # <--- MODIFIED: 使用修改后的 collate_fn
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=letterbox_collate_fn)
    coco_gt = val_dataset.coco # 这仍然有效，因为我们继承自 CocoDetection

    # --- 2. Load the model ---
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model_state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    
    # ↓↓↓ ================== 关键修改 ================== ↓↓↓
    # 你必须在这里实例化你 *新* 的多序列模型。
    # YoloBody 必须是能够接受 (BS, N, C, H, W) 输入的新模型类。
    # 
    # 例如:
    # from nets.yolo_multi_frame_model import YoloBodyMultiFrame
    # model = YoloBodyMultiFrame(num_classes=len(CLASS_NAMES), phi=PHI)
    # 
    # (这里我暂时保留 YoloBody，你需要确保它指向正确的类)
    model = YoloBodySST(num_classes=len(CLASS_NAMES), num_frame=SEQ_LEN)
    # ↑↑↑ ================== 关键修改 ================== ↑↑↑
    
    # weights_init(model) # 通常在评估时不需要
    model.load_state_dict(checkpoint) # <--- MODIFIED: 确保加载的是 'model_state_dict'
    model = model.to(device)
    print("Model loaded.")

    # --- 3. Call the evaluation function ---
    # (此部分无需修改)
    coco_evaluator = get_coco_map(
        model=model,
        dataloader=val_dataloader,
        coco_gt=coco_gt,
        device=device,
        confidence=CONFIDENCE,
        nms_iou=NMS_IOU
    )

    # --- 4. Print the results from the returned object ---
    # (此部分无需修改)
    if coco_evaluator:
        print("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35)
        coco_evaluator.summarize()
        print_per_class_results(coco_evaluator, CLASS_NAMES)
        
        map_50_95 = coco_evaluator.stats[0]
        print(f"\nReturned mAP @[IoU=0.50:0.95]: {map_50_95:.4f}")