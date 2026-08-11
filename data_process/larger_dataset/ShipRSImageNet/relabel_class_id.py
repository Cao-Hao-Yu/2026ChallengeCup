import os

# ================= 配置区域 =================

# 标签文件路径
hbb_folder = r'D:\DeepLearning\Challenger\data\LargerDataset\ShipRSImageNet\hbblabels'
obb_folder = r'D:\DeepLearning\Challenger\data\LargerDataset\ShipRSImageNet\obblabels'

CLASS_MAP = {
    '1': '0:-1',    # Ship (本身)
    '2': '0:-1',    # Warship
    '3': '0:3',    # Submarine
    '4': '0:0',    # Aircraft Carrier (包含 Enterprise, Nimitz, Midway)
    '5': '0:0',    # Enterprise -> 归入 Aircraft Carrier
    '6': '0:0',    # Nimitz -> 归入 Aircraft Carrier
    '7': '0:0',    # Midway -> 归入 Aircraft Carrier
    
    '9': '0:2',    # Destroyer
    '10': '0:2',   # Atago -> 归入 Destroyer
    '11': '0:2',   # Arleigh Burke -> 归入 Destroyer
    '12': '0:2',   # Hatsuyuki -> 归入 Destroyer
    '13': '0:2',   # Hyuga -> 归入 Destroyer
    '14': '0:2',   # Asagiri -> 归入 Destroyer
    
    '8': '0:2',    # Cruiser (原数据标注为 Ticonderoga，归入新小类或作为Destroyer同级)
    
    '15': '0:2',   # Frigate
    '16': '0:2',   # Perry -> 归入 Frigate
    
    '17': '0:3',   # Patrol
    
    '18': '0:1',   # Landing
    '19': '0:1',   # YuTing -> 归入 Landing
    '20': '0:1',   # YuDeng -> 归入 Landing
    '21': '0:1',   # YuDao -> 归入 Landing
    '22': '0:1',   # YuZhao -> 归入 Landing
    '23': '0:1',   # Austin -> 归入 Landing
    '24': '0:1',   # Osumi -> 归入 Landing
    '25': '0:1',   # Wasp -> 归入 Landing
    '26': '0:1',   # LSD_41 -> 归入 Landing
    '27': '0:1',   # LHA -> 归入 Landing
    
    '28': '0:3',   # Commander (根据层级图在Warship下，暂定为独立小类9)
    
    '29': '0:3',  # Auxiliary Ships
    '30': '0:3',  # Medical ship -> 归入 Auxiliary
    '31': '0:3',  # Test ship -> 归入 Auxiliary
    '32': '0:3',  # Training ship -> 归入 Auxiliary
    '33': '0:3',  # AOE -> 归入 Auxiliary
    '34': '0:3',  # Masyuu -> 归入 Auxiliary
    '35': '0:3',  # Sanantonio -> 归入 Auxiliary
    '36': '0:3',  # EPF -> 归入 Auxiliary
    
    # --- Merchant 系列 (小类 11-21) ---
    '37': '0:3',  # Merchant
    '38': '0:3',  # Container Ship
    '39': '0:3',  # RoRo
    '40': '0:3',  # Cargo
    '41': '0:3',  # Barge
    '42': '0:3',  # Tugboat
    '43': '0:3',  # Ferry
    '44': '0:3',  # Yacht
    '45': '0:3',  # Sailboat
    '46': '0:3',  # Fishing Vessel
    '47': '0:3',  # Oil Tanker
    '48': '0:3',  # Hovercraft
    '49': '0:3',  # Motorboat
}

# 2. 需要忽略的类别
IGNORE_CLASSES = {
    '50'
}

# ===========================================

def check_unmapped_classes(folder_path, class_map, ignore_set):
    """
    扫描文件夹中的所有标签文件，找出既未映射也未忽略的类别ID
    """
    unmapped_ids = set()
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        original_id = parts[0]
                        if original_id not in class_map and original_id not in ignore_set:
                            unmapped_ids.add(original_id)
        except Exception as e:
            print(f"读取文件 {filename} 出错: {e}")
            
    return unmapped_ids

def update_label_file(folder_path, class_map, ignore_set):
    """
    更新指定文件夹下的所有标签文件
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    updated_count = 0
    ignored_count = 0
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        new_lines = []
        needs_update = False
        file_ignored = 0
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                original_id = parts[0]
                
                # 逻辑判断
                if original_id in ignore_set:
                    file_ignored += 1
                    needs_update = True
                    continue
                
                if original_id in class_map:
                    parts[0] = class_map[original_id]
                    needs_update = True
                else:
                    # 既不在映射也不在忽略，保留原样并报警
                    print(f"警告: 文件 {filename} 中发现未处理的ID: {original_id}，已保留原ID。")
                
                new_lines.append(' '.join(parts))
            
            if needs_update:
                # 如果过滤后没有标签了，写入空文件（或直接写入空字符串）
                with open(filepath, 'w', encoding='utf-8') as f:
                    if new_lines:
                        f.write('\n'.join(new_lines) + '\n')
                    else:
                        f.write('')
                updated_count += 1
                ignored_count += file_ignored
                
        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {e}")
            
    return updated_count, ignored_count

def main():
    print("=" * 50)
    print("开始执行类别映射检查与转换程序")
    print("=" * 50)
    
    # 检查路径是否存在
    if not os.path.exists(hbb_folder):
        print(f"错误: HBB路径不存在 {hbb_folder}")
        return
    if not os.path.exists(obb_folder):
        print(f"错误: OBB路径不存在 {obb_folder}")
        return

    # 1. 检查 HBB 文件夹
    print(f"\n[1/4] 正在检查 HBB 路径: {hbb_folder}")
    hbb_unmapped = check_unmapped_classes(hbb_folder, CLASS_MAP, IGNORE_CLASSES)
    
    # 2. 检查 OBB 文件夹
    print(f"[2/4] 正在检查 OBB 路径: {obb_folder}")
    obb_unmapped = check_unmapped_classes(obb_folder, CLASS_MAP, IGNORE_CLASSES)
    
    # 合并未映射的ID
    all_unmapped = hbb_unmapped.union(obb_unmapped)
    
    if all_unmapped:
        print("\n[错误] 发现未映射且未声明忽略的类别 ID！请补充配置：")
        for uid in sorted(list(all_unmapped)):
            print(f"  - 缺少配置: '{uid}'")
        print("\n程序已暂停，请修改 CLASS_MAP 配置后重新运行。")
        return
    
    print("[检查通过] 所有类别 ID 均已在配置中定义。")
    
    # 3. 执行更新
    print("\n[3/4] 正在更新 HBB 标签...")
    hbb_files, hbb_ignored = update_label_file(hbb_folder, CLASS_MAP, IGNORE_CLASSES)
    print(f"  -> 已更新 {hbb_files} 个文件，忽略了 {hbb_ignored} 个目标。")
    
    print("[4/4] 正在更新 OBB 标签...")
    obb_files, obb_ignored = update_label_file(obb_folder, CLASS_MAP, IGNORE_CLASSES)
    print(f"  -> 已更新 {obb_files} 个文件，忽略了 {obb_ignored} 个目标。")
    
    print("\n转换完成！")

if __name__ == '__main__':
    main()
