import os
import shutil

# ====================== 修改为你的数据集根目录 ======================
root = r"D:\=0=DLProject\TEST_LEVIR_CD"
target_root = os.path.join(root, "samples")
# ===================================================================

target_A = os.path.join(target_root, "A")
target_B = os.path.join(target_root, "B")
target_label = os.path.join(target_root, "label")
list_dir = os.path.join(target_root, "list")

os.makedirs(target_A, exist_ok=True)
os.makedirs(target_B, exist_ok=True)
os.makedirs(target_label, exist_ok=True)
os.makedirs(list_dir, exist_ok=True)

splits = ["train", "val", "test"]

for split in splits:
    src_A = os.path.join(root, split, "A")
    src_B = os.path.join(root, split, "B")
    src_label = os.path.join(root, split, "OUT")

    if not all(os.path.isdir(p) for p in [src_A, src_B, src_label]):
        print(f"警告：{split} 的 A/B/OUT 目录缺失，跳过该集合")
        continue

    img_names = []
    for fname in os.listdir(src_A):
        if fname.lower().endswith(".png"):
            img_names.append(fname)

    new_name_list = []
    for fname in img_names:
        # 添加 train_ / val_ / test_ 前缀
        new_fname = f"{split}_{fname}"
        new_name_list.append(new_fname)

        shutil.copy2(os.path.join(src_A, fname), os.path.join(target_A, new_fname))
        shutil.copy2(os.path.join(src_B, fname), os.path.join(target_B, new_fname))
        shutil.copy2(os.path.join(src_label, fname), os.path.join(target_label, new_fname))

    # 生成txt，写入带前缀文件名
    txt_path = os.path.join(list_dir, f"{split}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for name in new_name_list:
            f.write(name + "\n")

    print(f"[{split}] 样本数量：{len(img_names)}，列表保存至 {txt_path}")

print("\n===== 数据集转换完成，和示例结构完全一致 =====")
print(f"{root}/samples/")
print("├─ A/")
print("├─ B/")
print("├─ label/")
print("└─ list/")
print("   ├─ train.txt")
print("   ├─ val.txt")
print("   └─ test.txt")
print("文件名样例：train_36_0512_0512.png，val_27_0000_0256.png，test_2_0000_0000.png")
print("原始数据集目录不会被修改。")
