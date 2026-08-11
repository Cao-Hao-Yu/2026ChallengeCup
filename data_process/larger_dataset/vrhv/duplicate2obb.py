## 将[x, y, w, h]标注改成[x1, y1][x2, y2][x3, y3][x4, y4]
import os

def hbb_to_obb(hbb_dir, obb_dir):
    """
    将YOLO格式的HBB标注转换为OBB标注。
    
    Args:
        hbb_dir (str): HBB标注文件路径 (输入路径)
        obb_dir (str): OBB标注文件保存路径 (输出路径)
    """
    
    # 检查输入路径是否存在
    if not os.path.exists(hbb_dir):
        print(f"错误：输入路径不存在 - {hbb_dir}")
        return

    # 如果输出路径不存在，则创建
    if not os.path.exists(obb_dir):
        os.makedirs(obb_dir)
        print(f"创建输出目录: {obb_dir}")

    print(f"开始转换: {hbb_dir} -> {obb_dir}")
    
    count_files = 0
    
    # 遍历所有标注文件
    for filename in os.listdir(hbb_dir):
        if filename.endswith('.txt'):
            hbb_path = os.path.join(hbb_dir, filename)
            obb_path = os.path.join(obb_dir, filename)
            
            obb_lines = []
            
            try:
                with open(hbb_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    
                    # HBB格式: class_id x_center y_center width height
                    # 索引:      0         1        2      3      4
                    if len(parts) < 5:
                        continue
                    
                    # 提取类别（保留 "1:-1" 格式）
                    class_id = parts[0]
                    
                    # 提取坐标信息并转为浮点数
                    xc = float(parts[1])
                    yc = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    # -----------------------------------------
                    # 核心转换逻辑：计算矩形四个顶点坐标
                    # -----------------------------------------
                    # 计算半宽和半高
                    half_w = w / 2
                    half_h = h / 2
                    
                    # 计算四个顶点坐标
                    # 左上
                    x1 = xc - half_w
                    y1 = yc - half_h
                    # 右上
                    x2 = xc + half_w
                    y2 = yc - half_h
                    # 右下
                    x3 = xc + half_w
                    y3 = yc + half_h
                    # 左下
                    x4 = xc - half_w
                    y4 = yc + half_h
                    
                    # 格式化输出字符串，保留类别ID和8个坐标值
                    # 格式: class_id x1 y1 x2 y2 x3 y3 x4 y4
                    obb_line = f"{class_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n"
                    obb_lines.append(obb_line)
                
                # 写入新的OBB文件
                with open(obb_path, 'w', encoding='utf-8') as f:
                    f.writelines(obb_lines)
                    
                count_files += 1
                
            except Exception as e:
                print(f"处理文件出错 {filename}: {e}")

    print(f"转换完成！共处理了 {count_files} 个文件。")
    print(f"结果已保存至: {obb_dir}")

if __name__ == "__main__":
    # 输入路径（上一步处理好的HBB标签路径）
    input_path = r"D:\DeepLearning\Challenger\data\LargerDataset\VHRV\hbblabels"
    
    # 输出路径（新的OBB标签路径）
    output_path = r"D:\DeepLearning\Challenger\data\LargerDataset\VHRV\obblabels"
    
    # 执行转换
    hbb_to_obb(input_path, output_path)
