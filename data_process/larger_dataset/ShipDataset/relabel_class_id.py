import os

# ================= 配置区域 =================

# 数据集标签路径
INPUT_FOLDER = r'D:\DeepLearning\Challenger\data\LargerDataset\ShipDataset\labels'
OUTPUT_FOLDER = r'D:\DeepLearning\Challenger\data\LargerDataset\ShipDataset\relabeled_result'

# 类别名称到ID的映射字典
# 格式：'类别名称': '层级标签'
# 逻辑参考：
#   0:0 -> Aircraft Carrier (航母)
#   0:1 -> Landing (登陆舰)
#   0:2 -> Destroyer/Frigate/Cruiser (驱护巡等主战舰艇)
#   0:3 -> Auxiliary/Merchant/Submarine (辅助/民船/潜艇)
#   -1   -> 忽略

CLASS_MAP = {
    # --- 航母 (0:0) ---
    'Nimitz Aircraft Carrier': '0:0',
    '001-aircraft carrier': '0:0',
    'Kitty Hawk class aircraft carrier': '0:0',
    'Forrestal-class Aircraft Carrier': '0:0',
    
    # --- 登陆舰 (0:1) ---
    'Whidbey Island-class dock landing ship': '0:1',
    'San Antonio-class amphibious transport dock': '0:1',
    '074-landing ship': '0:1',
    '072III-landing ship': '0:1',
    'Wasp-class amphibious assault ship': '0:1',
    '072II-landing ship': '0:1',
    '072A-landing ship': '0:1',
    '073-landing ship': '0:1',
    'Tarawa-class amphibious assault ship': '0:1',
    '071-amphibious transport dock': '0:1',
    'Lewis B. Puller-class expeditionary mobile base ship': '0:1',
    'Osumi-class landing ship': '0:1',
    'JMSDF LCU-2001 class utility landing crafts': '0:1',
    '074A-landing ship': '0:1',
    
    # --- 主战舰艇 - 驱护巡 (0:2) ---
    'Arleigh Burke-class Destroyer': '0:2',
    'Ticonderoga-class cruiser': '0:2',
    '054A-frigate': '0:2',
    'Oliver Hazard Perry-class frigate': '0:2',
    '056-corvette': '0:2',
    '051-destroyer': '0:2',
    '053H3-frigate': '0:2',
    '053H2G-frigate': '0:2',
    '052C-destroyer': '0:2',
    '052D-destroyer': '0:2',
    '053H1G-frigate': '0:2',
    'Sovremenny-class destroyer': '0:2',
    'Iowa-class battle ship': '0:2',
    'Asagiri-class Destroyer': '0:2',
    'Hatsuyuki-class destroyer': '0:2',
    'Takanami-class destroyer': '0:2',
    '054-frigate': '0:2',
    'Abukuma-class destroyer escort': '0:2',
    'Kongo-class destroyer': '0:2',
    '052-destroyer': '0:2',
    '051C-destroyer': '0:2',
    '052B-destroyer': '0:2',
    'Hatakaze-class destroyer': '0:2',
    'Akizuki-class destroyer': '0:2',
    'Zumwalt-class destroyer': '0:2',
    '051B-destroyer': '0:2',
    '055-destroyer': '0:2',
    'Izumo-class helicopter destroyer': '0:2',
    'Hyuga-class helicopter destroyer': '0:2',
    'Murasame-class destroyer': '0:2',
    
    # --- 潜艇 (0:3) ---
    'Submarine': '0:3',

    'Other Warship': '0:-1',    # ID 17
    'unknown': '0:-1',          # ID 11
    
    # --- 辅助/民船/其他 --
    'Barge': '0:3',
    'Towing vessel': '0:3',
    'Barracks Ship': '0:3',
    'Bunker': '0:3',
    'Sand Carrier': '0:3',
    '037-submarine chaser': '0:3',
    'Fishing Vessel': '0:3',
    '529-Minesweeper': '0:3',
    'unknown auxiliary ship': '0:3',
    '636-hydrographic survey ship': '0:3',
    'Avenger-class mine countermeasures ship': '0:3',
    'Traffic boat': '0:3',
    '081-Minesweeper': '0:3',
    'Tuzhong Class Salvage Tug': '0:3',
    'Independence-class littoral combat ship': '0:3',
    '639A-Hydroacoustic measuring ship': '0:3',
    'Bulk carrier': '0:3',
    'YO-25 class yard oiler': '0:3',
    '272-icebreaker': '0:3',
    '082II-Minesweeper': '0:3',
    'Henry J. Kaiser-class replenishment oiler': '0:3',
    'Lewis and Clark-class dry cargo ship': '0:3',
    'Yacht': '0:3',
    'Tank ship': '0:3',
    'Freedom-class littoral combat ship': '0:3',
    'YG-203 class yard gasoline oiler': '0:3',
    '037II-missile boat': '0:3',
    '815A-spy ship': '0:3',
    'Emory S. Land-class submarine tender': '0:3',
    'Sugashima-class minesweepers': '0:3',
    'YW-17 Class Yard Water': '0:3',
    '648-submarine repair ship': '0:3',
    '926-submarine support ship': '0:3',
    '037-hospital ship': '0:3',
    'Mercy-class hospital ship': '0:3',
    '815-spy ship': '0:3',
    'Hayabusa-class guided-missile patrol boats': '0:3',
    'Powhatan-class tugboat': '0:3',
    '903A-replenishment ship': '0:3',
    'Hiuchi-class auxiliary multi-purpose support ship': '0:3',
    '721-transport boat': '0:3',
    '920-hospital ship': '0:3',
    '909-experimental ship': '0:3',
    'USNS Spearhead': '0:3',
    'Uwajima-class minesweepers': '0:3',
    'Yaeyama-class minesweeper': '0:3',
    '922A-Salvage lifeboat': '0:3',
    'Uraga-class Minesweeper Tender': '0:3',
    'JS Chihaya': '0:3',
    '635-hydrographic Survey Ship': '0:3',
    'Cyclone-class patrol ship': '0:3',
    '909A-experimental ship': '0:3',
    '925-Ocean salvage lifeboat': '0:3',
    'Towada-class replenishment oilers': '0:3',
    '904-general stores issue ship': '0:3',
    '903-replenishment ship': '0:3',
    'Northampton-class tug': '0:3',
    'Xu Xiake barracks ship': '0:3',
    'Hatsushima-class minesweeper': '0:3',
    'USNS Montford Point': '0:3',
    'Blue Ridge class command ship': '0:3',
    '917-lifeboat': '0:3',
    '679-training ship': '0:3',
    'North Transfer 990': '0:3',
    '625C-Oceanographic Survey Ship': '0:3',
    '891A-training ship': '0:3',
    '904B-general stores issue ship': '0:3',
    'Hibiki-class ocean surveillance ships': '0:3',
    'Futami-class hydro-graphic survey ships': '0:3',
    'Kurobe-class training support ship': '0:3',
    'Tenryu-class training support ship': '0:3',
    'USNS Bob Hope': '0:3',
    'Mashu-class replenishment oilers': '0:3',
    'JS Suma': '0:3',
    '908-replenishment ship': '0:3',
    'JS Kurihama': '0:3',
    '680-training ship': '0:3',
    '901-fast combat support ship': '0:3',
    'Container Ship': '0:3',
    '022-missile boat': '0:3',
    '905-replenishment ship': '0:3',
    'Sacramento-class fast combat support ship': '0:3',
}

# 需要忽略的类别名称 (这些行将被删除)
IGNORE_CLASSES = {
}

# ===========================================

def process_dataset(input_dir, output_dir, class_map, ignore_set):
    """
    读取原标签，转换格式与ID，写入新路径
    """
    if not os.path.exists(input_dir):
        print(f"错误: 输入路径不存在 {input_dir}")
        return

    # 如果输出目录不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
    print(f"发现 {len(files)} 个标签文件，开始处理...\n")

    # 预扫描：检查未映射类别
    print("[1/2] 正在检查类别映射完整性...")
    unmapped_classes = set()
    for filename in files:
        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 9:
                    class_name = parts[8]
                    if class_name not in class_map and class_name not in ignore_set:
                        unmapped_classes.add(class_name)
    
    if unmapped_classes:
        print("\n[错误] 发现未定义映射的类别名称！请补充 CLASS_MAP：")
        for name in sorted(list(unmapped_classes)):
            print(f"  - '{name}': '0:X',")
        print("\n程序已暂停，请修改配置后重试。")
        return
    print("  -> 检查通过，所有类别均已定义。\n")

    # 执行转换
    print("[2/2] 正在转换并写入新标签...")
    total_processed = 0
    total_ignored = 0
    
    for filename in files:
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        
        new_lines = []
        file_ignored_count = 0
        
        with open(in_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 解析原格式: x1,y2,...,x4,y4,ClassName,Difficulty
            parts = line.split(',')
            
            # 必须包含至少10个字段 (8坐标 + 1类别 + 1难度)
            if len(parts) < 10:
                continue
                
            coords = parts[:8]
            class_name = parts[8]
            # difficulty = parts[9] # 丢弃
            
            if class_name in ignore_set:
                file_ignored_count += 1
                continue
            
            # 获取新ID
            new_id = class_map.get(class_name)
            if new_id is None:
                # 理论上预扫描已过滤，此处为保险
                continue
            
            # 构造新行: ID + 空格 + 坐标字符串
            # 示例: 0:3 299.0,389.0,246.0,395.0,207.0,67.0,259.0,60.0
            new_line = f"{new_id} {','.join(coords)}"
            new_lines.append(new_line)
        
        # 写入新文件
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines) + '\n')
            
        total_processed += 1
        total_ignored += file_ignored_count

    print(f"\n处理完成！")
    print(f"  - 处理文件数: {len(files)}")
    print(f"  - 忽略目标数: {total_ignored}")
    print(f"  - 新标签已保存至: {output_dir}")

def main():
    print("="*50)
    print("数据集标签重标注工具")
    print("="*50)
    process_dataset(INPUT_FOLDER, OUTPUT_FOLDER, CLASS_MAP, IGNORE_CLASSES)

if __name__ == '__main__':
    main()