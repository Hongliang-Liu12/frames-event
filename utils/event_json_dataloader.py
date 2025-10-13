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
    def __init__(self, json_path, image_root, input_shape, num_classes, epoch_length, mosaic, train,
                 augment_ration=0.9, mosaic_prob=0.5, mixup_prob=0.5, hsv_prob=0.5,
                 degrees=10.0, translate=0.1, shear=2.0):
        super(EventJsonDataset, self).__init__()
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.image_root = image_root
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.train = train

        self.augment_ration = augment_ration
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.hsv_prob = hsv_prob
        self.degrees = degrees
        self.translate = translate
        self.shear = shear

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
        self.step_now = -1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        self.step_now += 1
        
        do_augment = self.train and self.step_now < self.epoch_length * self.augment_ration

        if self.mosaic and self.rand() < self.mosaic_prob and do_augment:
            selected_ids = sample(self.image_ids, 3)
            selected_ids.append(self.image_ids[index])
            shuffle(selected_ids)
            image, box = self.get_random_data_with_Mosaic(selected_ids, self.input_shape)
        else:
            image, box = self.get_random_data(self.image_ids[index], self.input_shape, random=do_augment)

        if self.rand() < self.mixup_prob and len(box) > 0 and do_augment:
            mixup_id = sample(self.image_ids, 1)[0]
            mixup_image, mixup_box = self.get_random_data(mixup_id, self.input_shape, random=True)
            image, box = self.mixup(image, box, mixup_image, mixup_box)

        # 最终处理：确保为 float32, 并转置为 CHW
        image = np.array(image, dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))
        box = np.array(box, dtype=np.float32)

        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

        return image, box

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_image_and_boxes(self, image_id):
        img_info = self.images[image_id]
        img_path = os.path.join(self.image_root, img_info['file_name'])
        image = Image.open(img_path)
        image = cvtColor(image)
        boxes = self.annotations.get(image_id, [])
        return image, np.array(boxes)

    def get_random_data(self, image_id, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        image, box = self.get_image_and_boxes(image_id)
        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (114, 114, 114))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            if len(box) > 0:
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
            return image_data, box

        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (114, 114, 114))
        new_image.paste(image, (dx, dy))
        image = new_image
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image_data = np.array(image, np.uint8)
        if self.rand() < self.hsv_prob:
            hue_val = self.rand(-hue, hue) * 360
            sat_val = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
            val_val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
            x = cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV).astype(np.float32)
            x[..., 0] = (x[..., 0] + hue_val) % 360
            x[..., 1] = np.clip(x[..., 1] * sat_val, 0, 255)
            x[..., 2] = np.clip(x[..., 2] * val_val, 0, 255)
            image_data = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_HSV2RGB)
        image_data = image_data.astype(np.float32)
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
        return image_data, box

    def get_affine_matrix(self, target_size, degrees, translate, scales, shear):
        twidth, theight = target_size
        angle = self.rand(-degrees, degrees)
        scale = self.rand(scales[0], scales[1])
        R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)
        M = np.ones([2, 3])
        shear_x = math.tan(self.rand(-shear, shear) * math.pi / 180)
        shear_y = math.tan(self.rand(-shear, shear) * math.pi / 180)
        M[0] = R[0] + shear_y * R[1]
        M[1] = R[1] + shear_x * R[0]
        translation_x = self.rand(-translate, translate) * twidth
        translation_y = self.rand(-translate, translate) * theight
        M[0, 2] = translation_x
        M[1, 2] = translation_y
        return M, scale

    def apply_affine_to_bboxes(self, targets, target_size, M):
        num_gts = len(targets)
        if num_gts == 0: return targets
        twidth, theight = target_size
        corner_points = np.ones((4 * num_gts, 3))
        corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(4 * num_gts, 2)
        corner_points = corner_points @ M.T
        corner_points = corner_points.reshape(num_gts, 8)
        corner_xs = corner_points[:, 0::2]
        corner_ys = corner_points[:, 1::2]
        new_bboxes = np.concatenate((corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))).reshape(4, num_gts).T
        new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
        new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)
        targets[:, :4] = new_bboxes
        return targets

    def random_affine(self, img, targets=(), target_size=(640, 640), scales=(0.1, 2)):
        M, scale = self.get_affine_matrix(target_size, self.degrees, self.translate, scales, self.shear)
        img = cv2.warpAffine(img, M, dsize=target_size, borderValue=(114, 114, 114))
        if len(targets) > 0:
            targets = self.apply_affine_to_bboxes(targets, target_size, M)
        return img, targets

    def get_random_data_with_Mosaic(self, image_ids, input_shape, hue=.1, sat=1.5, val=1.5):
        h, w = input_shape
        yc = int(self.rand(0.5 * h, 1.5 * h))
        xc = int(self.rand(0.5 * w, 1.5 * w))
        mosaic_labels = []
        mosaic_img = np.full((h * 2, w * 2, 3), 114, dtype=np.uint8)
        for i, img_id in enumerate(image_ids):
            image, box = self.get_image_and_boxes(img_id)
            iw, ih = image.size
            if self.rand() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if len(box) > 0: box[:, [0, 2]] = iw - box[:, [2, 0]]
            scale = min(w / iw, h / ih)
            resized_w, resized_h = int(iw * scale), int(ih * scale)
            image = image.resize((resized_w, resized_h), Image.BICUBIC)
            image = np.array(image, dtype=np.uint8)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - resized_w, 0), max(yc - resized_h, 0), xc, yc
                x1b, y1b, x2b, y2b = resized_w - (x2a - x1a), resized_h - (y2a - y1a), resized_w, resized_h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - resized_h, 0), min(xc + resized_w, w * 2), yc
                x1b, y1b, x2b, y2b = 0, resized_h - (y2a - y1a), min(resized_w, x2a - x1a), resized_h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - resized_w, 0), yc, xc, min(yc + resized_h, h * 2)
                x1b, y1b, x2b, y2b = resized_w - (x2a - x1a), 0, resized_w, min(resized_h, y2a - y1a)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + resized_w, w * 2), min(yc + resized_h, h * 2)
                x1b, y1b, x2b, y2b = 0, 0, min(resized_w, x2a - x1a), min(resized_h, y2a - y1a)
            mosaic_img[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw, padh = x1a - x1b, y1a - y1b
            if len(box) > 0:
                box[:, [0, 2]] = box[:, [0, 2]] * scale + padw
                box[:, [1, 3]] = box[:, [1, 3]] * scale + padh
                mosaic_labels.append(box)
        if len(mosaic_labels):
            mosaic_labels = np.concatenate(mosaic_labels, 0)
            np.clip(mosaic_labels[:, 0], 0, 2 * w, out=mosaic_labels[:, 0])
            np.clip(mosaic_labels[:, 1], 0, 2 * h, out=mosaic_labels[:, 1])
            np.clip(mosaic_labels[:, 2], 0, 2 * w, out=mosaic_labels[:, 2])
            np.clip(mosaic_labels[:, 3], 0, 2 * h, out=mosaic_labels[:, 3])
        mosaic_img, mosaic_labels = self.random_affine(mosaic_img, mosaic_labels, target_size=(w, h), scales=(0.5, 1.5))
        if self.rand() < self.hsv_prob:
            hue_val = self.rand(-hue, hue) * 360
            sat_val = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
            val_val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
            x = cv2.cvtColor(mosaic_img, cv2.COLOR_RGB2HSV).astype(np.float32)
            x[..., 0] = (x[..., 0] + hue_val) % 360
            x[..., 1] = np.clip(x[..., 1] * sat_val, 0, 255)
            x[..., 2] = np.clip(x[..., 2] * val_val, 0, 255)
            mosaic_img = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return mosaic_img.astype(np.float32), mosaic_labels

    def mixup(self, origin_img, origin_labels, mixup_img, mixup_labels):
        r = np.random.beta(8.0, 8.0)
        mixed_img = r * origin_img + (1.0 - r) * mixup_img
        if len(origin_labels) > 0 and len(mixup_labels) > 0:
            mixed_labels = np.concatenate((origin_labels, mixup_labels), axis=0)
        elif len(origin_labels) > 0:
            mixed_labels = origin_labels
        else:
            mixed_labels = mixup_labels
        return mixed_img.astype(np.float32), mixed_labels

def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes


if __name__ == '__main__':
    # ==================================================== #
    #                  测试代码 (保存图片版)
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
        from utils import cvtColor

    # --- 1. 修改这里的参数 ---
    json_path = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/annotations/train.json'  # 测试JSON文件路径
    image_root = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/data'  # 图像根目录
    output_dir = "dataloader_test_images" 
    
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

    dataset = EventJsonDataset(
        json_path=json_path,
        image_root=image_root,
        input_shape=input_shape,
        num_classes=num_classes,
        epoch_length=100,
        mosaic=True,
        train=True,
        augment_ration=1.0, 
        mosaic_prob=0.8,    
        mixup_prob=0.5,     
        hsv_prob=0.5,
        degrees=10.0,
        translate=0.1,
        shear=2.0,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=yolo_dataset_collate
    )

    # --- 可视化并保存函数 (已修正) ---
    def visualize_and_save_batch(images_batch, bboxes_batch, save_path, class_names):
        """将一个批次的图像和边界框可视化并保存为单个文件。"""
        rows = int(np.ceil(np.sqrt(len(images_batch))))
        cols = int(np.ceil(len(images_batch) / rows))
        plt.figure(figsize=(cols * 6, rows * 6))
        
        for i in range(len(images_batch)):
            ax = plt.subplot(rows, cols, i + 1)
            
            # 从dataloader获取的图像是 CHW, float32, 范围 0-255
            img = images_batch[i]
            
            # 1. 从 CHW 转为 HWC
            img = img.transpose(1, 2, 0)
            # 2. 将 float32 [0, 255] 转换为 uint8 [0, 255] 以便正确显示
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            plt.imshow(img)
            ax.set_title(f'Sample {i+1}')
            
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

    # --- 执行测试 ---
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