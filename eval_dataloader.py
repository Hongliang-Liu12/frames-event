import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置matplotlib中文字体，确保标注可以正确显示中文
# 注意：在无GUI的服务器上，可能需要确保字体文件存在或matplotlib配置正确
# 如果服务器没有SimHei字体，可能会回退到默认字体
# plt.rcParams["font.sans-serif"] = ["SimHei"] 
# plt.rcParams["axes.unicode_minus"] = False

# 定义类别名称，根据EvDET200K数据集的10个类别
CLASS_NAMES = [
    "people", "car", "bicycle", "electric bicycle", 
    "basketball", "ping_pong", "goose", "cat", "bird", "UAV"
]

class Evdet200kCocoDataset(CocoDetection):
    """
    一个基础的、只加载单张图片的数据集。
    其标注 (target) 格式与YOLOX官方实现保持一致。
    """
    def __init__(self, root_dir, split="test"):
        """
        初始化数据集。
        
        参数:
            root_dir (str): 数据集根目录。
            split (str): 数据集划分，可选值为 "train", "val", "test"。
        """
        annotation_path = os.path.join(root_dir, "Event_Frame", "annotations", f"{split}.json")
        images_root = os.path.join(root_dir, "Event_Frame", "data")
        
        super().__init__(root=images_root, annFile=annotation_path)
    
    def __getitem__(self, index):
        """
        根据索引获取一个数据样本（单张图片和YOLOX格式的标注）。
        """
        img_id = self.ids[index]
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = os.path.join(self.root, img_info['file_name'])
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 将标注组织成 (N, 5) 的Numpy数组
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            box = [x, y, x + w, y + h]
            label = ann['category_id'] - 1
            labels.append(box + [label])
        
        if not labels:
            labels = np.empty((0, 5), dtype=np.float32)
        else:
            labels = np.array(labels, dtype=np.float32)
        
        # 图像处理: HWC -> CHW, 并归一化到[0, 1]
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return image_tensor, labels

    def __len__(self):
        return len(self.ids)

def Evdet200k_collate_fn(batch):
    """
    自定义的collate_function，用于将数据打包成YOLOX所需的格式。
    """
    images, labels = zip(*batch)
    
    images_batch = torch.stack(images, 0)
    
    if labels:
        max_num_labels = max(len(l) for l in labels)
        
        if max_num_labels == 0:
            labels_batch = torch.zeros((len(labels), 0, 5))
        else:
            labels_batch = torch.zeros((len(labels), max_num_labels, 5))
            
            for i, l in enumerate(labels):
                if len(l) > 0:
                    labels_batch[i, :len(l), :] = torch.from_numpy(l)
    
    return images_batch, labels_batch


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # --- 路径设置 ---
    root_dir = r"/home/lhl/Git/datasets/EvDET200K"
    output_dir = "test_eval_dataloader"
    
    # 自动创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")

    print(f"数据集根目录: {root_dir}")
    
    # 1. 创建数据集实例
    dataset = Evdet200kCocoDataset(root_dir, split="test")
    
    if len(dataset) == 0:
        print("错误：数据集为空！请检查路径和JSON文件是否正确。")
        exit()
        
    print(f"成功创建数据集，共包含 {len(dataset)} 张图片。")
    
    # 2. 创建DataLoader实例
    dataloader = DataLoader(
        dataset,
        batch_size=4, 
        shuffle=True,
        num_workers=0,
        collate_fn=Evdet200k_collate_fn
    )
    
    print(f"\n开始从DataLoader中读取数据并生成可视化结果到 '{output_dir}' 文件夹...")
    
    # 3. 循环处理前5个批次的数据
    num_batches_to_visualize = 5
    for i, (images_batch, targets_batch) in enumerate(dataloader):
        if i >= num_batches_to_visualize:
            break
            
        print(f"\n--- 正在处理批次 {i+1}/{num_batches_to_visualize} ---")
        print(f"图像批次形状: {images_batch.shape}")
        print(f"标签批次形状: {targets_batch.shape}")

        # 4. 可视化该批次的第一张图和其标注
        image = images_batch[0]
        target = targets_batch[0] 
        valid_targets = target[target[:, :4].sum(dim=1) > 0]

        print(f"批次中第一张图的有效边界框数量: {valid_targets.shape[0]}")
        
        img_to_show = image.numpy().transpose(1, 2, 0)
        
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img_to_show)
        ax.axis('off')
        
        for row in valid_targets:
            box = row[:4]
            label = int(row[4])
            
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='cyan', facecolor='none')
            ax.add_patch(rect)
            
            class_name = CLASS_NAMES[label]
            ax.text(x_min, y_min - 10, class_name, color='black', fontsize=12, 
                    bbox=dict(facecolor='cyan', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

        plt.suptitle("visualization", fontsize=16)
        
        # 5. 保存图像而不是显示
        save_path = os.path.join(output_dir, f"visualization_batch_{i}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig) # 关闭图像以释放内存
        print(f"可视化结果已保存至: {save_path}")

    print(f"\n数据加载和可视化测试完成，共生成 {num_batches_to_visualize} 张图片。")