import json
import os
from random import sample, shuffle

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

# from .utils import cvtColor, preprocess_input
from utils import cvtColor, preprocess_input

class EventJsonDataset(Dataset):
    def __init__(self, json_path, image_root, input_shape, num_classes, epoch_length, mosaic, train, augment_ration=0,mosaic_prob=0, hsv_prob=0):
        super(EventJsonDataset, self).__init__()
        # 读取并解析JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.image_root = image_root
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.train = train

        # 添加新的概率参数
        self.augment_ration = augment_ration
        self.hsv_prob = hsv_prob
        self.mosaic_prob = mosaic_prob


        
        # 准备图像和注释数据
        self.images = {img['id']: img for img in self.data['images']}
        self.annotations = {img_id: [] for img_id in self.images.keys()}
        
        # 按图像ID组织注释
        for ann in self.data['annotations']:
            img_id = ann['image_id']
            # COCO格式: [x,y,width,height] -> 转换为 [x1,y1,x2,y2,class_id]
            x, y, width, height = ann['bbox']
            x1, y1 = x, y
            x2, y2 = x + width, y + height
            class_id = ann['category_id']-1
            self.annotations[img_id].append([x1, y1, x2, y2, class_id])
        
        # 创建图像ID列表用于索引
        self.image_ids = list(self.images.keys())
        self.length = len(self.image_ids)
        self.step_now = -1
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        index = index % self.length
        self.step_now += 1
        
        # 训练时进行数据的随机增强，验证时不进行
        if self.mosaic:
            if self.rand() < self.mosaic_prob and self.step_now < self.epoch_length * self.augment_ration * self.length:
                # 随机选择3个额外的图像ID和当前ID一起进行Mosaic增强
                selected_indices = sample(self.image_ids, 3)
                selected_indices.append(self.image_ids[index])
                shuffle(selected_indices)
                image, box = self.get_random_data_with_Mosaic(selected_indices, self.input_shape)
            else:
                image, box = self.get_random_data(self.image_ids[index], self.input_shape, random=self.train)
        else:
            image, box = self.get_random_data(self.image_ids[index], self.input_shape, random=self.train)
        
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        
        return image, box
    
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
    
    def get_image_and_boxes(self, image_id):
        # 获取图像路径
        img_info = self.images[image_id]
        img_path = os.path.join(self.image_root, img_info['file_name'])
        
        # 读取图像并转换为RGB
        image = Image.open(img_path)
        image = cvtColor(image)
        
        # 获取对应的边界框
        boxes = self.annotations[image_id]
        
        return image, np.array(boxes)
    
    def get_random_data(self, image_id, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        # 获取图像和边界框
        image, box = self.get_image_and_boxes(image_id)
        
        # 获取图像尺寸
        iw, ih = image.size
        h, w = input_shape
        
        if not random or self.step_now > self.epoch_length * self.augment_ration * self.length:
            # 不进行随机增强，保持原始比例调整大小
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            
            # 调整图像大小并添加灰条
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            
            # 调整边界框
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # 过滤无效框
            
            return image_data, box
        

        # 进行随机增强
        # 缩放和长宽扭曲
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        
        # 添加灰条
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        
        # 翻转图像
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)      

        # 色域扭曲，添加概率控制
        if self.rand() < self.hsv_prob:
            hue = self.rand(-hue, hue)
            sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
            val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
            x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
            x[..., 0] += hue * 360
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:, :, 0] > 360, 0] = 360
            x[:, :, 1:][x[:, :, 1:] > 1] = 1
            x[x < 0] = 0
            image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        else:
            image_data = np.array(image, np.float32)
        
        # 调整边界框
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
    
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                
                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                
                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                
                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                
                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        
        return merge_bbox
    
    def get_random_data_with_Mosaic(self, image_ids, input_shape, max_boxes=100, hue=.1, sat=1.5, val=1.5):
        h, w = input_shape
        min_offset_x = self.rand(0.25, 0.75)
        min_offset_y = self.rand(0.25, 0.75)
        
        nws = [int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1))]
        nhs = [int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1))]
        
        place_x = [int(w * min_offset_x) - nws[0], int(w * min_offset_x) - nws[1], int(w * min_offset_x), int(w * min_offset_x)]
        place_y = [int(h * min_offset_y) - nhs[0], int(h * min_offset_y), int(h * min_offset_y), int(h * min_offset_y) - nhs[3]]
        
        image_datas = []
        box_datas = []
        index = 0
        
        for img_id in image_ids:
            # 获取图像和边界框
            image, box = self.get_image_and_boxes(img_id)
            
            # 图像大小
            iw, ih = image.size
            
            # 是否翻转图片
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]
            
            nw = nws[index]
            nh = nhs[index]
            image = image.resize((nw, nh), Image.BICUBIC)
            
            # 将图片放置到对应位置
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)
            
            index += 1
            box_data = []
            # 对边界框进行处理
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)
        
        # 合并图像
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)
        
        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]
        
        # 色域变换，添加概率控制
        if self.rand() < self.hsv_prob:
            hue = self.rand(-hue, hue)
            sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
            val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
            x = cv2.cvtColor(np.array(new_image / 255, np.float32), cv2.COLOR_RGB2HSV)
            x[..., 0] += hue * 360
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:, :, 0] > 360, 0] = 360
            x[:, :, 1:][x[:, :, 1:] > 1] = 1
            x[x < 0] = 0
            new_image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        
        # 合并边界框
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)
        
        return new_image, new_boxes

# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes


if __name__ == '__main__':

    CLASS_NAMES = [
        "people", "car", "bicycle", "electric bicycle", 
        "basketball", "ping_pong", "goose", "cat", "bird", "UAV"
    ]

    import torch
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import os
    from torch.utils.data import DataLoader
    
    # 设置中文字体，防止matplotlib显示中文乱码
    plt.rcParams["font.family"] = ["SimHei"]
    
    # 测试参数设置
    json_path = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/annotations/test.json'  # 测试JSON文件路径
    image_root = '/home/lhl/Git/datasets/EvDET200K/Event_Frame/data'  # 图像根目录
    input_shape = [640, 640]  # 输入图像尺寸
    num_classes = 10  # 类别数量，根据实际情况修改
    epoch_length = 1  #  epoch长度
    batch_size = 2  # 批次大小
    
    # 创建数据集实例
    dataset = EventJsonDataset(
        json_path=json_path,
        image_root=image_root,
        input_shape=input_shape,
        num_classes=num_classes,
        epoch_length=epoch_length,
        mosaic=True,  # 启用马赛克增强
        train=True,  # 训练模式
        augment_ration=0.8,  # 整体数据马赛克比例
        mosaic_prob=0,  # 添加 mosaic 增强概率
        hsv_prob=0 # 添加 hsv 增强概率
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=yolo_dataset_collate
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"数据加载器批次数量: {len(dataloader)}")
    
    # 可视化函数
    def visualize_samples(images, bboxes, class_names=None):
        """可视化样本图像及其边界框"""
        # 使用已经定义的CLASS_NAMES列表，确保显示具体类别名称
        if class_names is None:
            class_names = CLASS_NAMES
        
        plt.figure(figsize=(12, 6))
        
        for i in range(len(images)):
            plt.subplot(1, len(images), i+1)
            # 将图像数据转换回原始格式
            img = images[i].transpose(1, 2, 0).copy()
            # ---------------------
            # 这是正确的反归一化代码
            # ---------------------
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            
            img = img * std + mean
            img = img * 255.0
            
            # 确保像素值在0-255范围内
            img = np.clip(img, 0, 255)
            img = img.astype(np.uint8)
            
            plt.imshow(img)
            plt.title(f'样本 {i+1}')
            
            
            # 绘制边界框
            ax = plt.gca()
            boxes = bboxes[i]
            if len(boxes) > 0:
                for box in boxes:
                    # 转换边界框格式: [x_center, y_center, width, height] -> [x1, y1, x2, y2]
                    x_center, y_center, width, height, class_id = box
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    # 创建矩形
                    rect = patches.Rectangle(
                        (x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # 添加类别标签
                    class_name = class_names[int(class_id)]
                    plt.text(x1, y1-10, class_name, color='r', fontsize=12, 
                             bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    # 测试数据加载和可视化
    try:
        # 加载第一个批次数据
        images, bboxes = next(iter(dataloader))
        print(f"批次图像形状: {images.shape}")
        print(f"批次边界框数量: {[len(box) for box in bboxes]}")
        
        # 可视化样本
        print("可视化样本图像及其边界框...")
        visualize_samples(images, bboxes)
        # img_to_show = images[0].transpose(1, 2, 0)
        # img_to_show = (img_to_show * 255).astype(np.uint8)
        # plt.figure()
        # plt.imshow(img_to_show)
        # plt.title('原始图像')
        # plt.show()

        
        # 测试单独获取一个样本
        print("测试单独获取一个样本...")
        image_id = dataset.image_ids[0]
        image, boxes = dataset.get_image_and_boxes(image_id)
        print(f"单张图像尺寸: {image.size}")
        print(f"单张图像边界框数量: {len(boxes)}")
        
        # 显示原始图像
        plt.figure()
        plt.imshow(image)
        plt.title('原始图像')
        plt.show()
        
        # 测试随机增强
        print("测试随机数据增强...")
        enhanced_image, enhanced_boxes = dataset.get_random_data(image_id, input_shape)
        print(f"增强后图像形状: {enhanced_image.shape}")
        print(f"增强后边界框数量: {len(enhanced_boxes)}")
        
        # 显示增强后的图像
        plt.figure()
        plt.imshow(enhanced_image.astype(np.uint8))
        plt.title('增强后的图像')
        plt.show()
        
        # 测试马赛克增强
        if dataset.mosaic:
            print("测试马赛克数据增强...")
            selected_indices = sample(dataset.image_ids, 3)
            selected_indices.append(image_id)
            shuffle(selected_indices)
            mosaic_image, mosaic_boxes = dataset.get_random_data_with_Mosaic(selected_indices, input_shape)
            print(f"马赛克图像形状: {mosaic_image.shape}")
            print(f"马赛克边界框数量: {len(mosaic_boxes)}")
            
            # 显示马赛克图像
            plt.figure()
            plt.imshow(mosaic_image.astype(np.uint8))
            plt.title('马赛克增强后的图像')
            plt.show()
            
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()