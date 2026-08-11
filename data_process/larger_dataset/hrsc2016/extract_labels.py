# HRSC2016官方标注xml文件包括hbb和obb
# obb为xywhθ格式
import os
import xml.etree.ElementTree as ET
import math

# ================= 配置路径 =================
# XML 标注文件所在路径
xml_folder = r'D:\DeepLearning\Challenger\data\LargerDataset\HRSC2016\labels'

# 输出路径
hbb_output_folder = r'D:\DeepLearning\Challenger\data\LargerDataset\HRSC2016\hbblabels'
obb_output_folder = r'D:\DeepLearning\Challenger\data\LargerDataset\HRSC2016\obblabels'
# ===========================================

def ensure_dir(directory):
    """如果目录不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def convert_hbb(box_xmin, box_ymin, box_xmax, box_ymax, img_w, img_h):
    """
    将 HBB (xmin, ymin, xmax, ymax) 转换为 YOLO 格式 (x_center, y_center, w, h) 并归一化
    """
    # 计算中心点和宽高
    x_center = (box_xmin + box_xmax) / 2.0
    y_center = (box_ymin + box_ymax) / 2.0
    width = box_xmax - box_xmin
    height = box_ymax - box_ymin
    
    # 归一化
    x_center /= img_w
    y_center /= img_h
    width /= img_w
    height /= img_h
    
    return x_center, y_center, width, height

def convert_obb(cx, cy, w, h, angle_rad, img_w, img_h):
    """
    将 OBB (cx, cy, w, h, angle_rad) 转换为 四点坐标 (x1,y1,...,x4,y4) 并归一化
    假设 angle_rad 是弧度，OpenCV 风格的角度定义（长边定义或几何中心旋转）
    """
    # 角度转换处理：通常 HRSC 的角度是弧度，OpenCV 使用的是度数，这里我们直接用弧度进行三角函数计算
    # 注意：HRSC 定义中，角度通常相对于水平方向
    
    # 计算 cos 和 sin
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    
    # 计算旋转后的向量
    # 矩形框相对于中心的四个角点向量: (±w/2, ±h/2)
    # 对应的四个点:
    # 1. (-w/2, -h/2)
    # 2. (w/2, -h/2)
    # 3. (w/2, h/2)
    # 4. (-w/2, h/2)
    
    # 旋转并平移回图像坐标系
    # x' = cx + (x_local * cos_a - y_local * sin_a)
    # y' = cy + (x_local * sin_a + y_local * cos_a)
    
    corners = []
    for x_local, y_local in [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]:
        x_rot = cx + x_local * cos_a - y_local * sin_a
        y_rot = cy + x_local * sin_a + y_local * cos_a
        
        # 归一化
        x_rot /= img_w
        y_rot /= img_h
        
        corners.extend([x_rot, y_rot])
        
    return corners

def parse_and_convert(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图像信息
        filename_obj = root.find('Img_FileName')
        if filename_obj is None:
            print(f"警告: {xml_path} 缺少 Img_FileName，跳过")
            return
        
        filename = filename_obj.text
        
        # 获取图像尺寸
        width = int(root.find('Img_SizeWidth').text)
        height = int(root.find('Img_SizeHeight').text)
        
        hbb_lines = []
        obb_lines = []
        
        objects = root.findall('HRSC_Objects/HRSC_Object')
        if not objects:
            # 没有目标对象
            return

        for obj in objects:
            # 提取 Class_ID
            class_id_elem = obj.find('Class_ID')
            if class_id_elem is None:
                continue
            class_id = class_id_elem.text
            
            # --- HBB 处理 ---
            xmin = float(obj.find('box_xmin').text)
            ymin = float(obj.find('box_ymin').text)
            xmax = float(obj.find('box_xmax').text)
            ymax = float(obj.find('box_ymax').text)
            
            x_c, y_c, w, h = convert_hbb(xmin, ymin, xmax, ymax, width, height)
            # 格式: class_id x_center y_center width height
            hbb_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
            
            # --- OBB 处理 ---
            # 提取旋转框信息
            cx = float(obj.find('mbox_cx').text)
            cy = float(obj.find('mbox_cy').text)
            w_r = float(obj.find('mbox_w').text)
            h_r = float(obj.find('mbox_h').text)
            ang = float(obj.find('mbox_ang').text) # 弧度
            
            corners_normalized = convert_obb(cx, cy, w_r, h_r, ang, width, height)
            
            # 格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
            # 保留6位小数
            obb_str = f"{class_id} " + " ".join([f"{v:.6f}" for v in corners_normalized])
            obb_lines.append(obb_str)
            
        # 写入 HBB 文件
        hbb_save_path = os.path.join(hbb_output_folder, filename + '.txt')
        with open(hbb_save_path, 'w') as f:
            f.write('\n'.join(hbb_lines))
            
        # 写入 OBB 文件
        obb_save_path = os.path.join(obb_output_folder, filename + '.txt')
        with open(obb_save_path, 'w') as f:
            f.write('\n'.join(obb_lines))
            
        print(f"已处理: {filename}")
        
    except Exception as e:
        print(f"处理文件 {xml_path} 时发生错误: {e}")

def main():
    # 确保输出目录存在
    ensure_dir(hbb_output_folder)
    ensure_dir(obb_output_folder)
    
    # 遍历 XML 文件夹
    files = os.listdir(xml_folder)
    xml_files = [f for f in files if f.endswith('.xml')]
    
    print(f"共发现 {len(xml_files)} 个 XML 文件...")
    
    for xml_file in xml_files:
        xml_path = os.path.join(xml_folder, xml_file)
        parse_and_convert(xml_path)
        
    print("转换完成！")

if __name__ == '__main__':
    main()
