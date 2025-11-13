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
from nets.yolo import YoloBody
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

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
        # BGR to RGB conversion can be useful if not handled elsewhere
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        return image_np, target, img_info

def letterbox_collate_fn(batch):
    images, targets, img_infos = zip(*batch)
    processed_images, ratios = [], []
    input_size = (640, 640)
    for img in images:
        img_h, img_w = img.shape[:2]
        scale = min(input_size[0] / img_h, input_size[1] / img_w)
        ratios.append(scale)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded_img = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
        padded_img[0:new_h, 0:new_w] = resized_img
        padded_img = padded_img.transpose((2, 0, 1))
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        processed_images.append(padded_img)
    images_batch = torch.from_numpy(np.array(processed_images))
    return images_batch, list(targets), list(img_infos), ratios

def print_per_class_results(coco_eval, class_names):
    eval_results = coco_eval.eval
    precisions = eval_results['precision']
    recalls = eval_results['recall']
    cat_ids = coco_eval.cocoGt.getCatIds()
    id_to_name = {cat['id']: cat['name'] for cat in coco_eval.cocoGt.loadCats(cat_ids)}
    
    results_data = []
    # Iterate through precision results for each category
    for k_idx, cat_id in enumerate(cat_ids):
        # AP @[IoU=0.50:0.95 | area=all | maxDets=100]
        p = precisions[:, :, k_idx, 0, 2]
        p = p[p > -1]
        ap = np.mean(p) * 100.0 if p.size > 0 else float('nan')
        # AR @[IoU=0.50:0.95 | area=all | maxDets=100]
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
#                  ↓↓↓ REFACTORED EVALUATION FUNCTION ↓↓↓
# ================================================================ #
def get_coco_map(model, dataloader, coco_gt, device, confidence=0.01, nms_iou=0.65):
    """
    Performs COCO evaluation on a given model and dataloader.
    This function is designed to be imported and used within a training loop.
    """
    model.eval()  # Set model to evaluation mode
    
    num_classes = len(CLASS_NAMES)
    input_shape = (640, 640)
    results = []
    strides = [8, 16, 32]
    hw = [(int(input_shape[0] / s), int(input_shape[1] / s)) for s in strides]
    
    print("Starting evaluation...")
    for images, _, img_infos, ratios in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            # Decode and fuse logic
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

        # Format results
        for batch_idx, output_per_image in enumerate(final_outputs):
            if output_per_image is not None:
                final_outputs_cpu = output_per_image.cpu().numpy()
                top_label = final_outputs_cpu[:, 6].astype('int32')
                top_conf = final_outputs_cpu[:, 4] * final_outputs_cpu[:, 5]
                top_boxes = final_outputs_cpu[:, :4]
                
                ratio = ratios[batch_idx]
                h, w = img_infos[batch_idx]['height'], img_infos[batch_idx]['width']
                
                top_boxes /= ratio
                top_boxes[:, [0, 2]] = np.clip(top_boxes[:, [0, 2]], 0, w)
                top_boxes[:, [1, 3]] = np.clip(top_boxes[:, [1, 3]], 0, h)
                
                for i, c in enumerate(top_label):
                    # Get the actual category ID from the ground truth mapping
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

    # --- Use COCO API to compute and return results ---
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    return coco_eval

# ================================================================ #
#                  ↓↓↓ STANDALONE EXECUTION EXAMPLE ↓↓↓
# ================================================================ #
if __name__ == "__main__":
    # --- Configuration for standalone run ---
    DATASET_ROOT_DIR = "/home/lhl/Git/datasets/EvDET200K"
    MODEL_PATH = "/home/lhl/Git/frames-event/logs/ep001.pth"
    PHI = 's'
    BATCH_SIZE = 4
    CONFIDENCE = 0.01
    NMS_IOU = 0.65
    CUDA = True

    # --- 1. Setup device and dataset ---
    device = torch.device('cuda' if CUDA and torch.cuda.is_available() else 'cpu')
    print("Loading dataset...")
    val_dataset = Evdet200kCocoDataset(DATASET_ROOT_DIR, split="test")
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, collate_fn=letterbox_collate_fn)
    coco_gt = val_dataset.coco

    # --- 2. Load the model ---
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model_state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    
    model = YoloBody(num_classes=len(CLASS_NAMES), phi=PHI)
    # weights_init(model)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    print("Model loaded.")



    # --- 3. Call the evaluation function ---
    coco_evaluator = get_coco_map(
        model=model,
        dataloader=val_dataloader,
        coco_gt=coco_gt,
        device=device,
        confidence=CONFIDENCE,
        nms_iou=NMS_IOU
    )

    # --- 4. Print the results from the returned object ---
    if coco_evaluator:
        print("\n" + "="*35 + " COCO EVALUATION SUMMARY " + "="*35)
        coco_evaluator.summarize()
        print_per_class_results(coco_evaluator, CLASS_NAMES)
        
        # You can access specific stats like this:
        map_50_95 = coco_evaluator.stats[0]
        print(f"\nReturned mAP @[IoU=0.50:0.95]: {map_50_95:.4f}")