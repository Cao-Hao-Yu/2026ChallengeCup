import os

CLASS_MAP = {
    '0:-1': '125',
    '0:0': '10',
    '0:1': '11',
    '0:2': '12',
    '0:3': '13',

    '1:-1': '225',
    '1:4': '24',
    '1:5': '25',
    '1:6': '26',
    '1:7': '27',
    '1:8': '28',
    '1:9': '29',
    '1:10': '210',
    '1:11': '211',
    '1:12': '212',
    '1:13': '213',
    '1:14': '214',
    '1:15': '215',
    '1:16': '216',
    '1:17': '217',
    '1:18': '218',
    '1:19': '219',
    '1:20': '220',
    '1:21': '221',
    '1:22': '222',
    '1:23': '223',

    '2:24': '324',
}

def convert_labels(label_folder):
    """
    转换指定文件夹下的所有标签文件
    """
    # 检查文件夹是否存在
    if not os.path.exists(label_folder):
        print(f"错误: 文件夹路径不存在 -> {label_folder}")
        return

    # 获取所有文件
    files = os.listdir(label_folder)
    print(f"正在处理文件夹: {label_folder}")
    print(f"共发现 {len(files)} 个文件...")

    processed_count = 0
    
    for filename in files:
        # 这里假设标签文件是 .txt 格式，如果是其他后缀请修改
        if not filename.endswith('.txt'):
            continue
        
        file_path = os.path.join(label_folder, filename)
        new_lines = []
        changed = False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # 分割数据：第一部分是标签，后面是坐标
                parts = line.split()
                original_label = parts[0]
                
                # 查找映射表
                if original_label in CLASS_MAP:
                    new_label = CLASS_MAP[original_label]
                    # 重构行数据：新标签 + 原坐标
                    new_line = f"{new_label} {' '.join(parts[1:])}"
                    new_lines.append(new_line + '\n')
                    
                    if original_label != new_label:
                        changed = True
                else:
                    # 如果在映射表中找不到，保留原样并打印警告
                    print(f"警告: 文件 {filename} 中发现未定义映射的标签 '{original_label}'，已保留原样。")
                    new_lines.append(line + '\n')

            # 只有当内容发生变化时才写入文件，减少IO操作
            if changed:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(new_lines)
                processed_count += 1

        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {e}")

    print(f"完成！共修改了 {processed_count} 个文件。")

# !?注注?!
# 如果使用N:n的标注格式 需要在ultralytics中做非常多的更改
# 于是我采用Nn的混合标注形式 通过注释断言语句跳过nc检查等nc相关语句
# 这里的代码用于将N:n转换成Nn 25代表小类缺失
if __name__ == "__main__":
    label_path = r"D:\DeepLearning\Challenger\data\new_dataset_yolo\labels\train"
    
    convert_labels(label_path)
