## 重新标注 hrplane只有飞机一个类别 因此全部重新标注为 1:-1
import os

def convert_yolo_labels(label_dir):
    """
    遍历指定目录下的所有YOLO标注文件，将class_id为0的项修改为指定格式。
    
    Args:
        label_dir (str): 标注文件所在的文件夹路径
    """
    
    # 检查路径是否存在
    if not os.path.exists(label_dir):
        print(f"错误：路径不存在 - {label_dir}")
        return

    print(f"开始处理路径: {label_dir}")
    
    count_files = 0
    count_lines = 0
    
    # 遍历目录下的所有文件
    for filename in os.listdir(label_dir):
        # 检查文件扩展名，通常YOLO标注文件为.txt
        if filename.endswith('.txt'):
            file_path = os.path.join(label_dir, filename)
            
            # 用于存储修改后的内容
            new_lines = []
            file_modified = False
            
            # 读取文件内容
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # 处理每一行
                for line in lines:
                    # 去除首尾空格
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 分割行数据
                    parts = line.split()
                    
                    # 检查第一个数字是否为0（原飞机类别）
                    # 假设格式正确，第一个元素为class_id
                    if parts[0] == '0':
                        # 构建新格式: "1:-1" + 原有的坐标信息
                        # parts[1:] 包含了 x_center, y_center, width, height
                        new_line = "1:-1 " + " ".join(parts[1:]) + "\n"
                        new_lines.append(new_line)
                        count_lines += 1
                        file_modified = True
                    else:
                        # 如果有其他类别的ID，保持原样（根据描述数据集只有飞机，此项为保险起见）
                        new_lines.append(line + "\n")
                
                # 如果文件内容有修改，则写回原文件
                if file_modified:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.writelines(new_lines)
                    count_files += 1
                    
            except Exception as e:
                print(f"处理文件出错 {filename}: {e}")

    print(f"处理完成！共修改了 {count_files} 个文件，涉及 {count_lines} 行标注数据。")

if __name__ == "__main__":
    # 目标路径
    target_path = r"D:\DeepLearning\Challenger\data\LargerDataset\HRPlanes\hbblabels"
    
    # 执行转换
    convert_yolo_labels(target_path)

