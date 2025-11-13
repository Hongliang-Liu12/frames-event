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
from nets.yolo_frames_net import YoloBodySST
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from loguru import logger # [!!] 导入 logger

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

class Evdet200kCocoDataset(CocoDetection):
    def __init__(self, root_dir, split="train", seq_len=3):
        annotation_path = os.path.join(root_dir, "Event_Frame", "annotations", f"{split}.json")
        images_root = os.path.join(root_dir, "Event_Frame", "data")
        super().__init__(root=images_root, annFile=annotation_path)
        
        # --- FIX: Store a reference to the original, complete COCO API object
        self.original_coco_loader = self.coco

        self.seq_len = seq_len
        
        logger.info(f"Building sequences for {split} set (seq_len={self.seq_len})...")
        groups = {}
        # Use the original loader to get all image IDs
        for img_id in self.original_coco_loader.getImgIds():
            img_info = self.original_coco_loader.loadImgs(img_id)[0]
            file_name = img_info.get('file_name', '')
            folder = os.path.dirname(file_name)
            groups.setdefault(folder, []).append((img_id, file_name))

        id_to_sequence = {}
        for folder, items in groups.items():
            try:
                items_sorted = sorted(items, key=lambda x: int(os.path.splitext(os.path.basename(x[1]))[0]))
            except Exception:
                items_sorted = sorted(items, key=lambda x: x[1])
            
            ids_sorted = [it[0] for it in items_sorted]
            
            for i in range(self.seq_len - 1, len(ids_sorted)):
                target_id = ids_sorted[i]
                seq_ids = ids_sorted[i - self.seq_len + 1 : i + 1]
                id_to_sequence[target_id] = seq_ids

        self.id_to_sequence = id_to_sequence
        
        original_id_count = len(self.original_coco_loader.getImgIds())
        # self.ids will become the list of valid TARGET frame IDs
        self.ids = [img_id for img_id in self.original_coco_loader.getImgIds() if img_id in self.id_to_sequence]
        logger.info(f"Filtered {original_id_count} -> {len(self.ids)} valid target frames.")
        
        # <--- 新增开始: 创建一个只包含有效目标帧的 coco_gt 对象 ---
        logger.info("Creating a filtered COCO ground truth object for evaluation...")
        
        # 1. 创建一个新的、空的 COCO 对象
        filtered_coco = COCO()
        
        # 2. 填充筛选后的图像信息
        # Use the original loader to get data for the filtered IDs
        filtered_coco.dataset['images'] = self.original_coco_loader.loadImgs(self.ids)
        
        # 3. 获取并填充与这些图像相关的标注信息
        ann_ids = self.original_coco_loader.getAnnIds(imgIds=self.ids)
        filtered_coco.dataset['annotations'] = self.original_coco_loader.loadAnns(ann_ids)
        
        # 4. 复制类别信息
        filtered_coco.dataset['categories'] = self.original_coco_loader.loadCats(self.original_coco_loader.getCatIds())
        
        # 5. 为新的 coco 对象创建索引，这对于评估至关重要
        filtered_coco.createIndex()
        
        # 6. 用这个新的、经过筛选的 coco 对象替换掉原来的
        # This is now the filtered GT object for evaluation
        self.coco = filtered_coco
        logger.info("Filtered COCO ground truth object created successfully.")
        # <--- 新增结束 ---

    def __getitem__(self, index):
        # This is the target frame ID, which exists in the filtered self.coco
        img_id = self.ids[index]
        target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
        target_img_info = self.coco.loadImgs(img_id)[0]
        
        # This sequence contains IDs that may NOT be in the filtered self.coco
        seq_ids = self.id_to_sequence[img_id]
        image_seq_np = []
        
        for frame_id in seq_ids:
            # --- FIX: Use the saved original loader to get info for ALL frames in the sequence
            frame_info = self.original_coco_loader.loadImgs(frame_id)[0]
            image_path = os.path.join(self.root, frame_info['file_name'])
            image_np = cv2.imread(image_path)
            
            if image_np is None:
                logger.info(f"Warning: Could not read image {image_path}. Using black frame.")
                h = target_img_info.get('height', 640)
                w = target_img_info.get('width', 640)
                image_np = np.zeros((h, w, 3), dtype=np.uint8)
            image_seq_np.append(image_np)

        return image_seq_np, target, target_img_info

# -------------------- [ 修改开始: letterbox_collate_fn ] --------------------
def letterbox_collate_fn(batch):
    images_seqs, targets, img_infos = zip(*batch)
    # --- [ 修改：增加 paddings 列表 ] ---
    processed_seqs, ratios, paddings = [], [], []
    input_size = (640, 640)

    for seq in images_seqs:
        processed_frames = []
        for i, img in enumerate(seq):
            img_h, img_w = img.shape[:2]
            scale = min(input_size[0] / img_h, input_size[1] / img_w)

            new_w, new_h = int(img_w * scale), int(img_h * scale)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            padded_img = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
            
            # --- [ 新增：计算居中所需的 padding ] ---
            pad_top = (input_size[0] - new_h) // 2
            pad_left = (input_size[1] - new_w) // 2

            # --- [ 修改：将图像放置在计算好的居中位置 ] ---
            padded_img[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized_img
            
            # --- [ 修改：将 ratio 和 padding 的保存移到此处 ] ---
            if i == len(seq) - 1:
                # 仅保存最后一个（目标）帧的 ratio 和 padding
                ratios.append(scale)
                paddings.append((pad_left, pad_top)) # <-- 保存 padding

            padded_img = padded_img.transpose((2, 0, 1))
            padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
            processed_frames.append(padded_img)
        processed_seqs.append(np.stack(processed_frames, axis=0))

    images_batch = torch.from_numpy(np.stack(processed_seqs, axis=0))
    # --- [ 修改：返回 paddings ] ---
    return images_batch, list(targets), list(img_infos), ratios, paddings
# -------------------- [ 修改结束: letterbox_collate_fn ] --------------------

def print_per_class_results(coco_eval, class_names):
    # ... (此函数保持不变)
    eval_results = coco_eval.eval
    precisions = eval_results['precision']
    # recall is not directly available per class in the same way
    cat_ids = coco_eval.cocoGt.getCatIds()
    id_to_name = {cat['id']: cat['name'] for cat in coco_eval.cocoGt.loadCats(cat_ids)}
    
    results_data = []
    logger.info("\n" + "="*50)
    logger.info(f"{'CLASS':<20} | {'AP @[.5:.95]':^20}")
    logger.info("-" * 50)
    for k_idx, cat_id in enumerate(cat_ids):
        # areaRng = 'all', maxDets = 100
        p = precisions[:, :, k_idx, 0, 2]
        p = p[p > -1]
        ap = np.mean(p) * 100.0 if p.size > 0 else float('nan')
        results_data.append((id_to_name.get(cat_id, "unknown"), ap))
        ap_str = f"{ap:.3f}" if not np.isnan(ap) else "---"
        logger.info(f"{id_to_name.get(cat_id, 'unknown'):<20} | {ap_str:^20}")
    logger.info("=" * 50)

# -------------------- [ 修改开始: get_coco_map ] --------------------
def get_coco_map(model, dataloader, coco_gt, device, confidence=0.01, nms_iou=0.65):
    model.eval()
    num_classes = len(CLASS_NAMES)
    input_shape = (640, 640)
    results = []
    strides = [8,16,32]
    hw = [(int(input_shape[0] / s), int(input_shape[1] / s)) for s in strides]
    
    logger.info("Starting evaluation...")
    # --- [ 修改：接收 letterbox_collate_fn 返回的 paddings ] ---
    for images, _, img_infos, ratios, paddings in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
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

            for batch_idx, output_per_image in enumerate(final_outputs):
                if output_per_image is not None:
                    final_outputs_cpu = output_per_image.cpu().numpy()
                    top_label = final_outputs_cpu[:, 6].astype('int32')
                    top_conf = final_outputs_cpu[:, 4] * final_outputs_cpu[:, 5]
                    top_boxes = final_outputs_cpu[:, :4]

                    # --- [ 修改：获取 padding 并应用正确的坐标逆变换 ] ---
                    ratio = ratios[batch_idx]
                    pad_left, pad_top = paddings[batch_idx] # <-- 获取 padding
                    h, w = img_infos[batch_idx]['height'], img_infos[batch_idx]['width']

                    # 1. 从 640x640 坐标系平移回 letterbox 内的坐标系 (减去 padding)
                    top_boxes[:, [0, 2]] -= pad_left
                    top_boxes[:, [1, 3]] -= pad_top
                    
                    # 2. 缩放回原始图像坐标系 (除以 ratio)
                    top_boxes /= ratio
                    
                    # 3. 裁剪到原始图像边界
                    top_boxes[:, [0, 2]] = np.clip(top_boxes[:, [0, 2]], 0, w)
                    top_boxes[:, [1, 3]] = np.clip(top_boxes[:, [1, 3]], 0, h)
                    # --- [ 修改结束 ] ---

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

    # 验证ID匹配情况（现在应该匹配了）
    gt_img_ids = set(coco_gt.getImgIds())
    dt_img_ids = set([res["image_id"] for res in results]) if results else set()
    logger.info(f"\nGround Truth image IDs: {len(gt_img_ids)}")
    logger.info(f"Detection image IDs: {len(dt_img_ids)}")
    logger.info(f"IDs in GT but not in DT: {len(gt_img_ids - dt_img_ids)}") # 应该为0或非常小

    if not results:
        logger.info("No detections were made. Cannot evaluate.")
        return None

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    return coco_eval
# -------------------- [ 修改结束: get_coco_map ] --------------------


if __name__ == "__main__":
    DATASET_ROOT_DIR = "/home/lhl/Git/datasets/EvDET200K"
    MODEL_PATH = "/home/lhl/Git/frames-event/logs/newtwostage/only2_perfect_resume/42_7_0.0002_0.00015_4_model_B_EMA_ep010_map-0.5260.pth"
    BATCH_SIZE = 4
    CONFIDENCE = 0.01
    NMS_IOU = 0.65
    CUDA = True
    SEQ_LEN = 3

    device = torch.device('cuda' if CUDA and torch.cuda.is_available() else 'cpu')
    logger.info("Loading dataset...")
    val_dataset = Evdet200kCocoDataset(DATASET_ROOT_DIR, split="test", seq_len=SEQ_LEN)
    
    # --- [ 注意：这里使用的 collate_fn 已经是我们修改后的 letterbox_collate_fn ] ---
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=letterbox_collate_fn)
    coco_gt = val_dataset.coco

# -------------------- [ 您的模型加载代码 (保持不变) ] --------------------
    logger.info(f"Loading model from {MODEL_PATH}...")
    # Robust checkpoint loader: try normal torch.load first, then
    # fall back to a safe unpickler that ignores missing mmengine
    # classes so we can still extract tensors/state_dicts.
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
    except ModuleNotFoundError as e:
        msg = str(e)
        if 'mmengine' in msg:
            logger.warning("torch.load failed due to missing 'mmengine' during unpickle. Trying safe fallback to ignore mmengine objects and extract state_dict...")
            import pickle, io, types

            def _safe_loads(b):
                # Custom Unpickler that returns simple placeholders for mmengine classes
                class SafeUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module.startswith('mmengine'):
                            # return a harmless placeholder (a simple function that does nothing)
                            return lambda *args, **kwargs: None
                        return super().find_class(module, name)

                return SafeUnpickler(io.BytesIO(b)).load()

            fake_pickle = types.SimpleNamespace(loads=_safe_loads)
            try:
                checkpoint = torch.load(MODEL_PATH, map_location=device, pickle_module=fake_pickle)
            except Exception:
                logger.exception("Safe fallback failed. Cannot load checkpoint.")
                raise
        else:
            raise
    
    model = YoloBodySST(num_classes=len(CLASS_NAMES), phi='s', num_frame=SEQ_LEN) # 确保phi参数与训练时一致

    # 1. (可选, 但推荐) 检查权重是否被包裹
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
        logger.info("   Checkpoint file detected. Extracted 'model' state_dict.")
    elif 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
        logger.info("   Checkpoint file detected. Extracted 'state_dict'.")

    # 2. 获取新模型的 state_dict
    model_state_dict = model.state_dict()
    
    # 3. 过滤 checkpoint，只保留匹配的键
    #     这就是能自动过滤掉 "total_ops" 的关键
    load_dict = {
        k: v for k, v in checkpoint.items()
        if k in model_state_dict and model_state_dict[k].shape == v.shape
    }
    
    # 4. 打印加载信息
    model_keys = set(model_state_dict.keys())
    loaded_keys = set(load_dict.keys())
    unloaded_keys = model_keys - loaded_keys
    
    logger.info(f"   {len(loaded_keys)} out of {len(model_keys)} layers were successfully matched for loading.")
    if unloaded_keys:
        logger.info(f"   Warning: {len(unloaded_keys)} keys in the model were NOT found in the checkpoint:")
        for key in sorted(list(unloaded_keys))[:5]:
             logger.info(f"     - {key}")
        if len(unloaded_keys) > 5: logger.info("     - ... (and more)")

    # 5. 更新并加载过滤后的 state_dict
    model_state_dict.update(load_dict)
    model.load_state_dict(model_state_dict)
    
    model = model.to(device)
    logger.info("Model loaded.")
# -------------------- [ 模型加载代码结束 ] --------------------
    
    logger.info("Model loaded.")
    logger.info("--- 训练后的融合门 (Fusion Gate) ---")
    logger.info(f"C3 Gate: {model.fusion_gate_c3.item()}")
    logger.info(f"C4 Gate: {model.fusion_gate_c4.item()}")
    logger.info(f"C5 Gate: {model.fusion_gate_c5.item()}")
    
    # --- [ 注意：这里调用的 get_coco_map 已经是我们修改后的版本 ] ---
    coco_evaluator = get_coco_map(
        model=model,
        dataloader=val_dataloader,
        coco_gt=coco_gt,
        device=device,
        confidence=CONFIDENCE,
        nms_iou=NMS_IOU
    )

    if coco_evaluator:
        logger.info("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35)
        coco_evaluator.summarize()
        print_per_class_results(coco_evaluator, CLASS_NAMES)
        map_50_95 = coco_evaluator.stats[0]
        logger.info(f"\nReturned mAP @[IoU=0.50:0.95]: {map_50_95:.4f}")