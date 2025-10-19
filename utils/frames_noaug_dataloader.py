import json
import os
import math
from random import sample, shuffle

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

# 这一行保持不变，因为它在作为模块导入时是正确的
# 当您独立运行此文件进行测试时，如果报错，请确保您的项目结构
# 或者临时将 'from .utils import cvtColor' 改为 'from utils import cvtColor'
from .utils import cvtColor


class EventJsonDataset(Dataset):
    def __init__(self, json_path, image_root, input_shape, num_classes, seq_len=3):
        """
        简化版的构造函数，只保留了必要的参数。
        所有与数据增强相关的参数 (mosaic, train, augment_ration等) 已被移除。
        """
        super(EventJsonDataset, self).__init__()
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.input_shape = input_shape
        self.num_classes = num_classes

        # --- 数据解析部分 (保持不变) ---
        # images: dict mapping image_id -> image info
        self.images = {img['id']: img for img in self.data['images']}
        # annotations: map image_id -> list of bboxes [x1,y1,x2,y2,class]
        self.annotations = {img_id: [] for img_id in self.images.keys()}

        for ann in self.data['annotations']:
            img_id = ann['image_id']
            x, y, width, height = ann['bbox']
            x1, y1 = x, y
            x2, y2 = x + width, y + height
            class_id = ann['category_id'] - 1
            self.annotations[img_id].append([x1, y1, x2, y2, class_id])

        # Build sequences per folder (sliding window). Each image's `file_name` is
        # expected to contain a folder path, e.g. "folder_x/0000.jpg". We group
        # images by the folder and sort by the basename (numeric ordering), then
        # construct sliding windows of length 3 (configurable via seq_len).
        # sequence length (number of frames per sample)
        self.seq_len = seq_len
        # Group images by directory of the file_name
        groups = {}
        for img_id, img_info in self.images.items():
            file_name = img_info.get('file_name', '')
            folder = os.path.dirname(file_name)
            groups.setdefault(folder, []).append((img_id, file_name))

        # Sort each group's entries by the basename number (e.g. 0000, 0001, ...)
        sequences = []
        for folder, items in groups.items():
            # items: list of (img_id, file_name)
            try:
                items_sorted = sorted(items, key=lambda x: int(os.path.splitext(os.path.basename(x[1]))[0]))
            except Exception:
                # fallback to lexicographic sort if conversion fails
                items_sorted = sorted(items, key=lambda x: x[1])

            ids_sorted = [it[0] for it in items_sorted]
            for i in range(0, len(ids_sorted) - self.seq_len + 1):
                seq_ids = ids_sorted[i:i + self.seq_len]
                sequences.append(seq_ids)

        self.sequences = sequences
        self.length = len(self.sequences)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        简化版的 __getitem__。
        移除了所有数据增强 (Mosaic, Mixup, Affine) 的逻辑。
        现在它只会直接加载并处理单张图片。
        """
        index = index % self.length

        # sequence of image ids for this sample
        seq_ids = self.sequences[index]

        images_seq = []
        target_box = np.array([])

        # Load each frame in the sequence. We only keep the annotations of the
        # last frame as the target (as requested).
        for i, img_id in enumerate(seq_ids):
            image_data, box = self.get_data(img_id, self.input_shape)

            # image_data: H,W,3 float32
            img_arr = np.array(image_data, dtype=np.float32)
            img_arr = np.transpose(img_arr, (2, 0, 1))  # C,H,W
            images_seq.append(img_arr)

            if i == len(seq_ids) - 1:
                # target for this sequence is the annotations of the last frame
                target_box = np.array(box, dtype=np.float32)

        # Stack frames -> (nums, C, H, W)
        if len(images_seq) > 0:
            images_seq = np.stack(images_seq, axis=0)
        else:
            images_seq = np.zeros((self.seq_len, 3, self.input_shape[0], self.input_shape[1]), dtype=np.float32)

        # Convert target_box from (x1,y1,x2,y2,class) to (cx,cy,w,h,class)
        if target_box.size != 0:
            # ensure shape (N,5)
            if target_box.ndim == 1:
                target_box = target_box.reshape(-1, target_box.shape[0])
            target_box = target_box.copy()
            target_box[:, 2:4] = target_box[:, 2:4] - target_box[:, 0:2]  # w,h
            target_box[:, 0:2] = target_box[:, 0:2] + target_box[:, 2:4] / 2  # cx,cy

        # Ensure empty target_box has shape (0,5) for downstream consistency
        if target_box.size == 0:
            target_box = np.zeros((0, 5), dtype=np.float32)

        return images_seq, target_box

    def get_image_and_boxes(self, image_id):
        """加载单个图像及其标注框 (此函数保持不变)"""
        img_info = self.images[image_id]
        img_path = os.path.join(self.image_root, img_info['file_name'])
        image = Image.open(img_path)
        image = cvtColor(image)
        boxes = self.annotations.get(image_id, [])
        return image, np.array(boxes)

    def get_data(self, image_id, input_shape):
        """
        加载并调整图像和标注框的尺寸，不进行随机数据增强。
        图像会按比例缩放以适应 input_shape，然后用灰色进行填充。
        这个函数是原 get_random_data 方法中 if not random: 分支的逻辑。
        """
        image, box = self.get_image_and_boxes(image_id)
        iw, ih = image.size
        h, w = input_shape

        # 计算缩放比例，以保持图像的原始长宽比
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        
        # 计算填充的偏移量
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        
        # 缩放图像并将其粘贴到灰色背景上
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (114, 114, 114))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)
        
        # 相应地调整边界框坐标
        if len(box) > 0:
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            
            # 确保边界框不会超出图像边界
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            
            # 过滤掉那些在缩放后变得过小的框
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
            
        return image_data, box

# --- 已删除的数据增强相关方法 ---
# rand()
# get_random_data_with_Mosaic()
# get_affine_matrix()
# apply_affine_to_bboxes()
# random_affine()
# mixup()


def yolo_dataset_collate(batch):
    """这个函数保持不变"""
    # batch: list of (images_seq, boxes)
    images = []
    bboxes = []
    for img_seq, box in batch:
        images.append(img_seq)
        bboxes.append(box)

    # stack into (bs, nums, c, h, w)
    images = np.stack(images, axis=0)
    return images, bboxes




# ===============================================================
#  测试 Dataset 的 Main 函数
# ===============================================================
if __name__ == "__main__":

    def draw_boxes(image_bgr, boxes_cxcywh, class_names):
        """ 在图像上绘制边界框 (cx, cy, w, h 格式)，并使用类别名称 """
        if boxes_cxcywh is None or len(boxes_cxcywh) == 0:
            return image_bgr
        img_with_boxes = image_bgr.copy()
        for box in boxes_cxcywh:
            cx, cy, w, h, class_id = box
            class_id = int(class_id)
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            color = (0, 255, 0) # Green
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            
            # 使用 class_names 列表获取标签
            if 0 <= class_id < len(class_names):
                label = class_names[class_id]
            else:
                label = f"ID: {class_id}" # 如果ID超出范围，则显示ID作为后备
            
            cv2.putText(img_with_boxes, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return img_with_boxes

    # --- 1. 参数设置 ---
    CLASS_NAMES = [
        "people", "car", "bicycle", "electric bicycle", 
        "basketball", "ping_pong", "goose", "cat", "bird", "UAV"
    ]

    train_json_path = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/annotations/train.json'
    image_root      = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/data'
    
    input_shape = (640, 640)
    num_classes = len(CLASS_NAMES) # 自动设置为10
    seq_len     = 3
    
    # 测试多少个样本
    num_test_samples = 4
    
    output_dir = "dataset_test_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"测试图像将保存在: {os.path.abspath(output_dir)}")
    
    # --- 2. 初始化 Dataset ---
    print("正在初始化 Dataset...")
    try:
        train_dataset = EventJsonDataset(
            json_path=train_json_path, image_root=image_root,
            input_shape=input_shape, num_classes=num_classes, seq_len=seq_len
        )
    except FileNotFoundError:
        print(f"错误: 找不到文件或目录。请检查路径:\n  JSON: {train_json_path}\n  Image Root: {image_root}")
        exit()

    if len(train_dataset) < num_test_samples:
        print(f"错误：数据集中的序列数 ({len(train_dataset)}) 少于要测试的样本数 ({num_test_samples})。")
        exit()
    
    print(f"Dataset 初始化成功，总序列数: {len(train_dataset)}。")
    print(f"将从数据集中随机挑选 {num_test_samples} 个样本进行组合。")

    # --- 3. 循环取出随机样本并处理 ---
    sample_rows = []
    # 从数据集中随机选择 num_test_samples 个不重复的索引
    random_indices = sample(range(len(train_dataset)), num_test_samples)

    for index in random_indices:
        images_seq, target_box = train_dataset[index]
        
        print(f"  正在处理随机样本索引 {index}...")
        
        # 存放3张原始图 + 1张标注图
        output_frames = [] 
        
        # 遍历原始序列中的每一帧
        for s_idx in range(images_seq.shape[0]):
            frame_chw = images_seq[s_idx]
            frame_hwc = np.transpose(frame_chw, (1, 2, 0))
            frame_uint8 = frame_hwc.astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            output_frames.append(frame_bgr)

        # 创建第4张图：即最后一张图的副本，并在上面画框
        if output_frames:
            last_frame_copy = output_frames[-1].copy()
            annotated_frame = draw_boxes(last_frame_copy, target_box, CLASS_NAMES)
            output_frames.append(annotated_frame)

            # 将这4张图水平拼接成一行
            combined_row = np.hstack(output_frames)
            sample_rows.append(combined_row)

    # --- 4. 将所有样本行垂直合并成一张大图并保存 ---
    if sample_rows:
        # 垂直堆叠所有样本行
        final_image = np.vstack(sample_rows)
        
        save_name = "all_samples_combined.jpg"
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, final_image)
        
        print("\n测试完成。所有随机样本已合并并保存到一张图中。")
        print(f"图片保存路径: {os.path.abspath(save_path)}")
    else:
        print("\n测试失败，未能生成任何图像。")