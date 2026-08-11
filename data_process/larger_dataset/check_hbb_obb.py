import os
import cv2
import numpy as np
import random

def visualize_conversion(hbb_dir, obb_dir, output_dir, img_size=(1166, 753), num_samples=5):
    """
    随机抽取样本，将HBB和OBB标签绘制在同一张图上进行对比。
    
    Args:
        hbb_dir (str): HBB标签文件夹路径
        obb_dir (str): OBB标签文件夹路径
        output_dir (str): 可视化结果保存路径
        img_size (tuple): 生成的背景图尺寸 (width, height)，默认(1166, 753)
        num_samples (int): 随机抽取检查的文件数量
    """
    
    if not os.path.exists(hbb_dir) or not os.path.exists(obb_dir):
        print("错误：标签路径不存在，请检查路径设置。")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有HBB文件列表
    all_files = [f for f in os.listdir(hbb_dir) if f.endswith('.txt')]
    
    # 随机抽取文件（如果文件数少于要求数量，则全选）
    sample_files = random.sample(all_files, min(num_samples, len(all_files)))
    
    print(f"开始可视化检查，共抽取 {len(sample_files)} 个样本...")
    print(f"当前模拟图像尺寸: 宽={img_size[0]}, 高={img_size[1]}")

    for filename in sample_files:
        hbb_path = os.path.join(hbb_dir, filename)
        obb_path = os.path.join(obb_dir, filename)
        
        if not os.path.exists(obb_path):
            print(f"警告：在OBB路径下找不到对应文件 {filename}，跳过。")
            continue

        # 1. 创建一个随机的背景图 (模拟复杂的背景)
        # 注意：cv2创建图像时 shape 顺序是，即
        img_h, img_w = img_size[1], img_size[0]
        
        # 生成带有随机噪点的灰色背景，模拟航拍图质感
        background = np.random.randint(100, 180, (img_h, img_w, 3), dtype=np.uint8)
        # 添加一些高斯噪声模糊
        background = cv2.GaussianBlur(background, (5, 5), 0)
        
        # ---------------------------------------------------
        # 2. 绘制 HBB (水平框) - 蓝色
        # ---------------------------------------------------
        with open(hbb_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5: continue
            
            # HBB格式: class x_center y_center width height
            # 注意：绘图时坐标需要乘以图片尺寸
            _, xc, yc, w, h = parts
            xc, yc, w, h = float(xc)*img_w, float(yc)*img_h, float(w)*img_w, float(h)*img_h
            
            # 转为左上角坐标 (x_min, y_min)
            x_min = int(xc - w/2)
            y_min = int(yc - h/2)
            x_max = int(xc + w/2)
            y_max = int(yc + h/2)
            
            # 绘制蓝色矩形 (线宽2)
            cv2.rectangle(background, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            
            # 标注 "HBB"
            cv2.putText(background, "HBB", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # ---------------------------------------------------
        # 3. 绘制 OBB (旋转框) - 红色
        # ---------------------------------------------------
        with open(obb_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9: continue # OBB至少需要 class + 8个坐标
            
            # OBB格式: class x1 y1 x2 y2 x3 y3 x4 y4
            # 注意：使用对应的宽高进行还原
            coords = [float(x)*img_w if i%2==0 else float(x)*img_h for i, x in enumerate(parts[1:9])]
            
            # 构造多边形点集 (需要int32类型)
            pts = np.array([
                [coords[0], coords[1]],
                [coords[2], coords[3]],
                [coords[4], coords[5]],
                [coords[6], coords[7]]
            ], dtype=np.int32)
            
            # 绘制红色多边形 (线宽2，方便看清)
            cv2.polylines(background, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            
            # 标注 "OBB"
            cv2.putText(background, "OBB", (int(coords[0]), int(coords[1])-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 保存结果
        save_path = os.path.join(output_dir, f"vis_{filename.replace('.txt', '.png')}")
        cv2.imwrite(save_path, background)
        print(f"已保存: {save_path}")

    print("检查完成！")

if __name__ == "__main__":
    # 输入路径配置
    hbb_path = r"D:\DeepLearning\Challenger\data\LargerDataset\ShipRSImageNet\hbblabels"
    obb_path = r"D:\DeepLearning\Challenger\data\LargerDataset\ShipRSImageNet\obblabels"
    
    # 输出路径配置
    output_path = r"D:\DeepLearning\Challenger\data\LargerDataset\obb_hbb_check"
    
    visualize_conversion(hbb_path, obb_path, output_path, img_size=(1280, 800), num_samples=10)
