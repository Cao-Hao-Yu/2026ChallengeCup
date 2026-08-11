import os
import xml.etree.ElementTree as ET
from pathlib import Path
import math

def get_rotated_box_points(cx, cy, w, h, angle_rad, img_w, img_h):
    """
    根据旋转框参数计算四个角点的归一化坐标
    """
    w = abs(w)
    h = abs(h)
    
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    w2 = w / 2
    h2 = h / 2
    
    # 计算未旋转时的四个角点相对中心点偏移
    corners_rel = [
        (-w2, -h2),
        ( w2, -h2),
        ( w2,  h2),
        (-w2,  h2)
    ]
    
    points_norm = []
    for rx, ry in corners_rel:
        # 应用旋转
        x_rot = rx * cos_a - ry * sin_a
        y_rot = rx * sin_a + ry * cos_a
        
        # 加上中心点偏移
        x_abs = x_rot + cx
        y_abs = y_rot + cy
        
        # 归一化
        x_norm = x_abs / img_w
        y_norm = y_abs / img_h
        
        # 截断防止越界
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        
        points_norm.extend([x_norm, y_norm])
        
    return points_norm

def parse_voc_xml_and_convert(xml_path, hbb_output_dir, obb_output_dir):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[Error] 解析XML失败: {xml_path}, {e}")
        return

    # 1. 获取图像尺寸
    size_elem = root.find('size')
    if size_elem is None:
        return
    
    try:
        img_w = int(size_elem.find('width').text)
        img_h = int(size_elem.find('height').text)
    except:
        return

    if img_w <= 0 or img_h <= 0:
        return

    # 获取文件名
    filename_elem = root.find('filename')
    img_filename = filename_elem.text if filename_elem is not None else Path(xml_path).stem
    txt_filename = Path(img_filename).stem + ".txt"
    
    hbb_lines = []
    obb_lines = []

    # 2. 遍历对象
    for obj in root.findall('object'):
        # --- 提取唯一的类别 ID ---
        # 优先级: level_3 (最细粒度) -> level_2 -> level_1 -> level_0
        # 对应你提供的数据：叶子节点都在 level_3 (如 Nimitz=6)，DOCK=50 这种可能在 level_2 或特殊字段
        class_id = None
        
        l3 = obj.find('level_3')
        l2 = obj.find('level_2')
        l1 = obj.find('level_1')
        
        if l3 is not None and l3.text != '0':
            class_id = l3.text
        elif l2 is not None:
            # 处理没有level_3的情况，例如 DOCK 可能定义在 level_2
            class_id = l2.text
        elif l1 is not None:
            class_id = l1.text
            
        if class_id is None:
            # 兜底：如果XML里一个level都没有，尝试读取name进行映射（极少情况）
            # 这里为了稳健性，如果没有ID则跳过或给默认值
            continue 

        # --- HBB 处理 ---
        bndbox = obj.find('bndbox')
        if bndbox is not None:
            try:
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # 容错：坐标反转
                if xmin > xmax: xmin, xmax = xmax, xmin
                if ymin > ymax: ymin, ymax = ymax, ymin
                
                if xmax > xmin and ymax > ymin:
                    x_center = ((xmin + xmax) / 2.0) / img_w
                    y_center = ((ymin + ymax) / 2.0) / img_h
                    width = (xmax - xmin) / img_w
                    height = (ymax - ymin) / img_h
                    
                    hbb_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            except:
                pass

        # --- OBB 处理 (带容错) ---
        obb_coords = []
        
        # 尝试 1: 读取 Polygon
        polygon = obj.find('polygon')
        valid_poly = False
        if polygon is not None:
            temp_pts = []
            for i in range(1, 5):
                x_tag = polygon.find(f'x{i}')
                y_tag = polygon.find(f'y{i}')
                if x_tag is not None and y_tag is not None:
                    try:
                        x = float(x_tag.text) / img_w
                        y = float(y_tag.text) / img_h
                        x = max(0.0, min(1.0, x))
                        y = max(0.0, min(1.0, y))
                        temp_pts.extend([x, y])
                    except:
                        pass
            
            if len(temp_pts) == 8:
                obb_coords = temp_pts
                valid_poly = True
        
        # 尝试 2: 如果 Polygon 无效，读取 Rotated Box 修正
        if not valid_poly:
            rot_box = obj.find('rotated_box')
            if rot_box is not None:
                try:
                    cx = float(rot_box.find('cx').text)
                    cy = float(rot_box.find('cy').text)
                    w = float(rot_box.find('width').text)
                    h = float(rot_box.find('height').text)
                    rot = float(rot_box.find('rot').text)
                    
                    obb_coords = get_rotated_box_points(cx, cy, w, h, rot, img_w, img_h)
                except:
                    pass

        if obb_coords:
            coord_str = " ".join([f"{c:.6f}" for c in obb_coords])
            obb_lines.append(f"{class_id} {coord_str}\n")

    # 3. 写入文件
    if hbb_lines:
        save_path = os.path.join(hbb_output_dir, txt_filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.writelines(hbb_lines)
            
    if obb_lines:
        save_path = os.path.join(obb_output_dir, txt_filename)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.writelines(obb_lines)

def batch_process_dataset(xml_dir, hbb_dir, obb_dir):
    os.makedirs(hbb_dir, exist_ok=True)
    os.makedirs(obb_dir, exist_ok=True)
    
    xml_files = [f for f in os.listdir(xml_dir) if f.endswith('.xml')]
    total = len(xml_files)
    
    print(f"开始处理: {total} 个文件...")
    
    for i, xml_file in enumerate(xml_files):
        xml_path = os.path.join(xml_dir, xml_file)
        parse_voc_xml_and_convert(xml_path, hbb_dir, obb_dir)
        if (i + 1) % 500 == 0:
            print(f"进度: {i+1}/{total}")
            
    print("处理完成。")

# ==========================================
# 配置路径并执行
# ==========================================

xml_annotations_dir = r"D:\DeepLearning\Challenger\data\LargerDataset\ShipRSImageNet\labels\VOC_Format\Annotations"
hbb_output_dir = r"D:\DeepLearning\Challenger\data\LargerDataset\ShipRSImageNet\hbblabels"
obb_output_dir = r"D:\DeepLearning\Challenger\data\LargerDataset\ShipRSImageNet\obblabels"

if __name__ == "__main__":
    batch_process_dataset(xml_annotations_dir, hbb_output_dir, obb_output_dir)
