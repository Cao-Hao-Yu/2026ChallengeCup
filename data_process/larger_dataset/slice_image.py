import os
import cv2
import numpy as np
import shutil

def slice_dataset(
    image_dir, 
    label_dir, 
    output_image_dir, 
    output_label_dir, 
    mode='hbb', 
    target_size=1280, 
    sub_sizes=[960, 640], 
    overlap_ratio=0.2, 
    min_area_ratio=0.5
):
    """
    对遥感数据集进行切片处理，支持HBB和OBB格式。
    修复了OBB模式下无法生成标签的问题，支持被截断的OBB（梯形/多边形）。
    """
    
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.tif'))]
    print(f"模式: {mode.upper()} | 开始处理 {len(image_files)} 张图像...")

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        # 尝试匹配 .txt 标签
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # 小图直接复制
        if w <= target_size and h <= target_size:
            cv2.imwrite(os.path.join(output_image_dir, img_name), img)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(output_image_dir, label_name))
            continue

        # 解析标签
        annotations = []
        if os.path.exists(label_path):
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    annotations.append(parts)

        # 获取切片坐标
        slices_coords = _get_slice_coords(w, h, sub_sizes, overlap_ratio)
        base_name = os.path.splitext(img_name)[0]
        
        for idx, (sx, sy, sw, sh) in enumerate(slices_coords):
            new_img_name = f"{base_name}__slice_{idx}.jpg"
            new_label_name = f"{base_name}__slice_{idx}.txt"
            
            # 保存切片图像
            crop_img = img[sy:sy+sh, sx:sx+sw]
            cv2.imwrite(os.path.join(output_image_dir, new_img_name), crop_img)
            
            new_labels = []
            
            for parts in annotations:
                # 处理 class_id (支持 "base:class" 格式)
                cls_id_raw = parts[0]
                
                # -----------------------------------------
                # HBB 模式处理 (逻辑不变)
                # -----------------------------------------
                if mode == 'hbb':
                    if len(parts) < 5: continue
                    
                    xc = float(parts[1]) * w
                    yc = float(parts[2]) * h
                    bw = float(parts[3]) * w
                    bh = float(parts[4]) * h
                    
                    x_min = xc - bw / 2
                    y_min = yc - bh / 2
                    x_max = xc + bw / 2
                    y_max = yc + bh / 2
                    
                    inter_x1 = max(sx, x_min)
                    inter_y1 = max(sy, y_min)
                    inter_x2 = min(sx + sw, x_max)
                    inter_y2 = min(sy + sh, y_max)
                    
                    inter_w = inter_x2 - inter_x1
                    inter_h = inter_y2 - inter_y1
                    
                    if inter_w <= 0 or inter_h <= 0: continue
                    
                    orig_area = bw * bh
                    inter_area = inter_w * inter_h
                    if inter_area / orig_area < min_area_ratio: continue
                    
                    new_xc = ((inter_x1 + inter_x2) / 2 - sx) / sw
                    new_yc = ((inter_y1 + inter_y2) / 2 - sy) / sh
                    new_bw = inter_w / sw
                    new_bh = inter_h / sh
                    
                    new_labels.append(f"{cls_id_raw} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}\n")

                # -----------------------------------------
                # OBB 模式处理 (核心修复)
                # -----------------------------------------
                elif mode == 'obb':
                    # 格式: class x1 y1 x2 y2 x3 y3 x4 y4
                    if len(parts) < 9: continue
                    
                    try:
                        # 1. 解析原始多边形坐标 (像素级)
                        # 严格按照 x1,y1, x2,y2 ... 顺序解析
                        points = np.array([
                            [float(parts[1])*w, float(parts[2])*h],
                            [float(parts[3])*w, float(parts[4])*h],
                            [float(parts[5])*w, float(parts[6])*h],
                            [float(parts[7])*w, float(parts[8])*h]
                        ], dtype=np.float32)
                        
                        # 计算原始面积
                        orig_area = cv2.contourArea(points)
                        if orig_area < 1.0: continue
                        
                        # 2. 使用 Sutherland-Hodgman 算法计算与切片矩形的交集
                        # 这样可以正确处理被截断成梯形、三角形的情况
                        roi_rect = (sx, sy, sx+sw, sy+sh) # x_min, y_min, x_max, y_max
                        inter_pts = _get_intersection_points(points, roi_rect)
                        
                        if len(inter_pts) == 0:
                            continue
                            
                        # 3. 计算交集面积并过滤
                        inter_area = cv2.contourArea(inter_pts)
                        
                        if orig_area == 0: continue
                        
                        area_ratio = inter_area / orig_area
                        if area_ratio < min_area_ratio:
                            continue
                        
                        # 4. 坐标归一化
                        # 保存交集多边形的顶点（可能是4个，也可能是5个、6个等）
                        # YOLO OBB 格式通常使用前4个点，但如果是为了保留截断形状，
                        # 这里将输出所有顶点。如果必须输出4个点，
                        # 应该使用 cv2.minAreaRect(inter_pts) 再次拟合，但这会损失形状精度。
                        # 下面的代码采用直接保存所有顶点的方式（更精确），
                        # 如果你必须使用矩形，请取消下方 minAreaRect 部分的注释。
                        
                        # --- 方案A：保存精确多边形顶点 (推荐，保留截断形状) ---
                        final_pts = []
                        # 如果顶点多于4个，且你的训练框架不支持，需要做凸包或minAreaRect
                        # 大多数 OBB 框架支持 4-8 个点。
                        # 为了兼容性，我们使用 minAreaRect 拟合回旋转矩形 (方案B)
                        
                        # --- 方案B：拟合成旋转矩形 (兼容性更好) ---
                        new_rect = cv2.minAreaRect(inter_pts)
                        new_box = cv2.boxPoints(new_rect)
                        
                        # 归一化
                        norm_pts = []
                        for p in new_box:
                            px = (p[0] - sx) / sw
                            py = (p[1] - sy) / sh
                            norm_pts.extend([px, py])
                        
                        line_str = f"{cls_id_raw} " + " ".join([f"{v:.6f}" for v in norm_pts]) + "\n"
                        new_labels.append(line_str)

                    except Exception as e:
                        print(f"处理OBB标签出错 {img_name}: {e}")
                        continue

            # 保存标签
            if new_labels:
                with open(os.path.join(output_label_dir, new_label_name), 'w', encoding='utf-8') as f:
                    f.writelines(new_labels)

    print("处理完成！")

def _get_intersection_points(poly_pts, rect):
    """
    使用 Sutherland-Hodgman 算法计算多边形与矩形的交集
    poly_pts: np.array (N, 2) 浮点数
    rect: (x_min, y_min, x_max, y_max)
    返回: 交集多边形的顶点数组
    """
    x_min, y_min, x_max, y_max = rect
    
    # 初始化输出为原始多边形顶点
    output = list(poly_pts)
    
    # 定义裁剪矩形的四条边 (左、右、下、上)
    # 注意图像坐标系 y轴向下
    # 每条边进行一次裁剪
    edges = [
        ('left', x_min),
        ('right', x_max),
        ('bottom', y_max), # y_max 是下边界
        ('top', y_min)
    ]
    
    for edge_type, edge_val in edges:
        if not output:
            break
        input_list = output
        output = []
        S = input_list[-1] # 上一个点
        
        for E in input_list:
            # 判断点是否在边内侧
            is_inside_E = False
            is_inside_S = False
            
            if edge_type == 'left':
                is_inside_E = E[0] >= edge_val
                is_inside_S = S[0] >= edge_val
            elif edge_type == 'right':
                is_inside_E = E[0] <= edge_val
                is_inside_S = S[0] <= edge_val
            elif edge_type == 'top': # y_min
                is_inside_E = E[1] >= edge_val
                is_inside_S = S[1] >= edge_val
            elif edge_type == 'bottom': # y_max
                is_inside_E = E[1] <= edge_val
                is_inside_S = S[1] <= edge_val

            if is_inside_E:
                if not is_inside_S:
                    # 从外到内，计算交点并添加
                    inter_pt = _compute_intersection(S, E, edge_type, edge_val)
                    if inter_pt: output.append(inter_pt)
                output.append(E) # 添加当前点
            elif is_inside_S:
                # 从内到外，计算交点并添加
                inter_pt = _compute_intersection(S, E, edge_type, edge_val)
                if inter_pt: output.append(inter_pt)
            
            S = E
            
    if len(output) < 3:
        return np.array([])
        
    return np.array(output, dtype=np.float32)

def _compute_intersection(p1, p2, edge_type, edge_val):
    """
    计算线段 p1-p2 与边界线的交点
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # 避免除以 0
    if x1 == x2 and edge_type in ['left', 'right']:
        return None
    if y1 == y2 and edge_type in ['top', 'bottom']:
        return None
        
    if edge_type == 'left' or edge_type == 'right':
        x = edge_val
        y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        return (x, y)
    else: # top or bottom
        y = edge_val
        x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
        return (x, y)

# 辅助函数：生成切片坐标 (保持不变)
def _get_slice_coords(img_w, img_h, sub_sizes, overlap):
    slices = []
    base_size = max(sub_sizes)
    stride = int(base_size * (1 - overlap))
    
    y_starts = [0]
    curr_y = 0
    while curr_y + base_size < img_h:
        curr_y += stride
        y_starts.append(curr_y)
    if y_starts[-1] + base_size < img_h:
        y_starts.append(img_h - base_size)
    
    x_starts = [0]
    curr_x = 0
    while curr_x + base_size < img_w:
        curr_x += stride
        x_starts.append(curr_x)
    if x_starts[-1] + base_size < img_w:
        x_starts.append(img_w - base_size)
        
    y_starts = sorted(list(set([max(0, y) for y in y_starts])))
    x_starts = sorted(list(set([max(0, x) for x in x_starts])))
    
    for sy in y_starts:
        for sx in x_starts:
            chosen_size = None
            for s in sorted(sub_sizes, reverse=True):
                if sy + s <= img_h and sx + s <= img_w:
                    chosen_size = s
                    break
            
            if chosen_size is None:
                min_s = min(sub_sizes)
                if sy + min_s <= img_h and sx + min_s <= img_w:
                    chosen_size = min_s
                else:
                    continue
            
            slices.append((sx, sy, chosen_size, chosen_size))
            
    return slices

if __name__ == "__main__":
    # 测试 OBB 切片
    slice_dataset(
        image_dir=r"D:\DeepLearning\Challenger\data\LargerDataset\HRPlanes\images",
        label_dir=r"D:\DeepLearning\Challenger\data\LargerDataset\HRPlanes\hbblabels",
        output_image_dir=r"D:\DeepLearning\Challenger\data\LargerDataset\slice_image_temp",
        output_label_dir=r"D:\DeepLearning\Challenger\data\LargerDataset\slice_image_temp",
        mode='hbb',
        target_size=1280,
        sub_sizes=[960, 640],
        overlap_ratio=0.2,
        min_area_ratio=0.2
    )
