import pickle

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def compare_pkl(file1, file2):
    obj1 = load_pkl(file1)
    obj2 = load_pkl(file2)

    if obj1 == obj2:
        print("两个 pkl 文件内容完全一致 ✅")
    else:
        print("两个 pkl 文件内容不同 ❌")

        # 如果是字典，可以打印 key 差异
        if isinstance(obj1, dict) and isinstance(obj2, dict):
            keys1, keys2 = set(obj1.keys()), set(obj2.keys())
            if keys1 != keys2:
                print("  键不同：")
                print("   仅在文件1:", keys1 - keys2)
                print("   仅在文件2:", keys2 - keys1)
            else:
                for k in keys1:
                    if obj1[k] != obj2[k]:
                        print(f"  键 {k} 对应的值不同")
        # 如果是列表，可以打印长度或前几个差异
        elif isinstance(obj1, list) and isinstance(obj2, list):
            if len(obj1) != len(obj2):
                print(f"  列表长度不同: {len(obj1)} vs {len(obj2)}")
            else:
                for i, (a, b) in enumerate(zip(obj1, obj2)):
                    if a != b:
                        print(f"  第 {i} 个元素不同: {a} vs {b}")
                        break  # 避免打印太多
        else:
            print("无法逐项详细比较（数据类型不同或太复杂）")

if __name__ == "__main__":
    file1 = '/home/ubuntu/custom-sparsebev/data/nuscenes/nuscenes_infos_train_sweep.pkl'
    file2 = '/home/ubuntu/BEVDet/2025-08-18/nuscenes_infos_train_sweep.pkl'
    compare_pkl(file1, file2)
