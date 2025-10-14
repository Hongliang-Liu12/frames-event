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
    def __init__(self, json_path, image_root, input_shape, num_classes):
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
        self.images = {img['id']: img for img in self.data['images']}
        self.annotations = {img_id: [] for img_id in self.images.keys()}

        for ann in self.data['annotations']:
            img_id = ann['image_id']
            x, y, width, height = ann['bbox']
            x1, y1 = x, y
            x2, y2 = x + width, y + height
            class_id = ann['category_id'] - 1
            self.annotations[img_id].append([x1, y1, x2, y2, class_id])

        self.image_ids = list(self.images.keys())
        self.length = len(self.image_ids)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        简化版的 __getitem__。
        移除了所有数据增强 (Mosaic, Mixup, Affine) 的逻辑。
        现在它只会直接加载并处理单张图片。
        """
        index = index % self.length
        
        # 直接获取数据，不进行任何随机增强
        image, box = self.get_data(self.image_ids[index], self.input_shape)

        # 最终处理：确保为 float32, 并转置为 CHW
        image = np.array(image, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))
        box = np.array(box, dtype=np.float32)

        # 将 box 格式从 (x1, y1, x2, y2) 转换为 (cx, cy, w, h)
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]  # 计算宽高
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2  # 计算中心点

        return image, box

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
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes


if __name__ == '__main__':
    # ==================================================== #
    #                  测试代码 (保存图片版)
    # ==================================================== #
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    # --- 修复导入问题 ---
    # 当直接运行此脚本时，相对导入会失败，我们用一个try-except来处理
    try:
        from .utils import cvtColor
    except ImportError:
        # 假设 utils.py 和这个脚本在同一个文件夹下
        # 如果您没有 utils.py 文件，可以先注释掉所有 cvtColor 调用
        # 并确保您的输入图片都是RGB格式
        from utils import cvtColor

    # --- 1. 修改这里的参数 ---
    json_path = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/annotations/train.json'  # 测试JSON文件路径
    image_root = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/data'  # 图像根目录
    output_dir = "dataloader_test_images_no_aug" 
    
    # --- 2. 根据你的数据集修改 ---
    num_classes = 10 
    CLASS_NAMES = [
        "people", "car", "bicycle", "electric bicycle", 
        "basketball", "ping_pong", "goose", "cat", "bird", "UAV"
    ]

    # --- 3. 其他测试参数 ---
    input_shape = [640, 640]
    batch_size = 4

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 修改了这里：使用简化版的 Dataset ---
    # 实例化数据集时不再需要传入数据增强相关的参数
    dataset = EventJsonDataset(
        json_path=json_path,
        image_root=image_root,
        input_shape=input_shape,
        num_classes=num_classes
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # 仍然可以打乱数据集的顺序
        num_workers=0,
        collate_fn=yolo_dataset_collate
    )

    # --- 可视化并保存函数 (与之前完全相同) ---
    def visualize_and_save_batch(images_batch, bboxes_batch, save_path, class_names):
        """将一个批次的图像和边界框可视化并保存为单个文件。"""
        rows = int(np.ceil(np.sqrt(len(images_batch))))
        cols = int(np.ceil(len(images_batch) / rows))
        plt.figure(figsize=(cols * 6, rows * 6))
        
        for i in range(len(images_batch)):
            ax = plt.subplot(rows, cols, i + 1)
            
            img = images_batch[i]
            img = img.transpose(1, 2, 0)
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            plt.imshow(img)
            ax.set_title(f'Sample {i+1} (No Augmentation)')
            
            bboxes = bboxes_batch[i]
            if len(bboxes) > 0:
                for box in bboxes:
                    x_center, y_center, w_box, h_box, class_id = box[0], box[1], box[2], box[3], int(box[4])
                    x1 = x_center - w_box / 2
                    y1 = y_center - h_box / 2
                    
                    color = plt.cm.get_cmap('hsv', num_classes)(class_id)
                    rect = plt.Rectangle((x1, y1), w_box, h_box, linewidth=2, edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1 - 5, class_names[class_id], bbox=dict(facecolor=color, alpha=0.5), color='white', fontsize=8)
            
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"测试图片已保存到: {save_path}")

    # --- 执行测试 (与之前完全相同) ---
    try:
        images, bboxes = next(iter(dataloader))
        print(f"Batch images shape: {images.shape}")
        
        save_path = os.path.join(output_dir, "dataloader_visualization.png")
        visualize_and_save_batch(images, bboxes, save_path, CLASS_NAMES)
        
    except StopIteration:
        print("数据加载器为空，无法生成测试图片。请检查数据集路径和内容。")
    except FileNotFoundError:
        print(f"错误: 找不到文件或目录, 请检查 json_path 和 image_root 是否正确:")
        print(f"  json_path: '{json_path}'")
        print(f"  image_root: '{image_root}'")
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()