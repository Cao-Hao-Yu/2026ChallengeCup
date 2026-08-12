import os
import cv2
import random
import numpy as np

def visualize_slice_check(slice_dir, output_dir, mode='hbb', num_samples=20):
    """
    随机抽取切片后的样本，绘制边界框并保存，用于检查切片质量。
    支持 HBB 和 OBB 模式切换。
    
    Args:
        slice_dir (str): 切片后的数据文件夹（包含图片和txt）
        output_dir (str): 可视化结果保存路径
        mode (str): 'hbb' 或 'obb'，选择检查模式
        num_samples (int): 随机抽取检查的样本数量
    """
    
    if not os.path.exists(slice_dir):
        print(f"错误：切片路径不存在 - {slice_dir}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 获取所有切片后的图片文件
    all_files = os.listdir(slice_dir)
    image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # 随机抽取样本（如果文件数少于要求数量，则全选）
    sample_files = random.sample(image_files, min(num_samples, len(image_files)))
    
    print(f"当前模式: {mode.upper()} | 开始可视化检查，共抽取 {len(sample_files)} 个切片样本...")

    for img_name in sample_files:
        img_path = os.path.join(slice_dir, img_name)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(slice_dir, label_name)
        
        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f"无法读取图像: {img_name}")
            continue
        
        h, w = img.shape[:2]
        
        # 检查是否有对应的标签文件
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                # 根据模式决定最小字段数
                min_len = 9 if mode == 'obb' else 5
                if len(parts) < min_len:
                    continue
                
                # 解析类别 ID (支持 "base:class" 格式，只取最后一个数字作为显示)
                cls_id_full = parts[0]
                cls_id_display = cls_id_full.split(':')[-1] if ':' in cls_id_full else cls_id_full
                
                # 根据类别生成固定颜色，方便区分不同目标
                # 使用哈希值保证同一个类别在不同图上颜色一致
                # random.seed(int(cls_id_display) if cls_id_display.isdigit() else hash(cls_id_display))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                
                # -----------------------------------------
                # HBB 模式处理
                # -----------------------------------------
                if mode == 'hbb':
                    # 格式: class_id x_center y_center width height
                    xc = float(parts[1])
                    yc = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])
                    
                    # 转换为像素坐标
                    x_center = int(xc * w)
                    y_center = int(yc * h)
                    width = int(bw * w)
                    height = int(bh * h)
                    
                    # 计算左上角和右下角
                    x_min = int(x_center - width / 2)
                    y_min = int(y_center - height / 2)
                    x_max = int(x_center + width / 2)
                    y_max = int(y_center + height / 2)
                    
                    # 绘制矩形
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
                    
                    # 绘制标签文字
                    cv2.putText(img, cls_id_display, (x_min, y_min - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # -----------------------------------------
                # OBB 模式处理
                # -----------------------------------------
                elif mode == 'obb':
                    # 格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    try:
                        # 解析 4 个角点坐标
                        coords = [float(p) for p in parts[1:9]]
                        
                        # 转换为像素坐标并构建多边形点集
                        points = []
                        for i in range(4):
                            px = int(coords[i*2] * w)
                            py = int(coords[i*2 + 1] * h)
                            points.append([px, py])
                        
                        # 转换为 numpy 数组格式 (4, 1, 2)
                        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                        
                        # 绘制多边形 (旋转框)
                        cv2.polylines(img, [points], isClosed=True, color=color, thickness=2)
                        
                        # 在第一个角点处绘制标签
                        text_pos = (points[0][0][0], points[0][0][1] - 5)
                        cv2.putText(img, cls_id_display, text_pos, 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                    
                    except Exception as e:
                        print(f"解析OBB坐标出错: {line.strip()} | 错误: {e}")
                        continue

        # 保存结果
        save_path = os.path.join(output_dir, f"vis_{img_name}")
        cv2.imwrite(save_path, img)

    print(f"检查完成！结果已保存至: {output_dir}")

if __name__ == "__main__":
    # 输入路径
    slice_path = r"D:\DeepLearning\Challenger\data\LargerDataset\slice_image_temp"
    
    # 输出路径
    vis_output_path = r"D:\DeepLearning\Challenger\data\LargerDataset\slice_image_check"

    # 如果模式选择错误不会有标签输出
    visualize_slice_check(slice_path, vis_output_path, mode='hbb', num_samples=20)
