import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2

def fuse_model(module):
    """
    再帰的に子モジュールを走査し、以下の条件を満たすモジュールに対して融合を試みる：
      - モジュールが nn.Sequential の場合、連続する Conv2d, BatchNorm2d, (ReLU) のグループを検出して融合
      - モジュールがカスタムモジュール（例: Conv）で、属性 'conv' と 'bn' を持つ場合、さらに属性 'act' が ReLU ならそれも含めて融合
    """
    # まず、子モジュールに対して再帰的に適用
    for name, child in module.named_children():
        fuse_model(child)
        # カスタムモジュールの場合：conv と bn を持つなら融合を試みる
        if hasattr(child, 'conv') and hasattr(child, 'bn'):
            fusion_group = ['conv', 'bn']
            if hasattr(child, 'act') and isinstance(child.act, nn.ReLU):
                fusion_group.append('act')
            try:
                torch.quantization.fuse_modules(child, fusion_group, inplace=True)
                print(f"Fused module {name} with group {fusion_group}")
            except Exception as e:
                print(f"Failed to fuse module {name} with group {fusion_group}: {e}")
        # nn.Sequentialの場合、Fusion対象のグループを検出して融合を試みる
        elif isinstance(child, nn.Sequential):
            fusion_groups = []
            i = 0
            while i < len(child):
                if (i + 1 < len(child) and 
                    isinstance(child[i], nn.Conv2d) and 
                    isinstance(child[i+1], nn.BatchNorm2d)):
                    group = [str(i), str(i+1)]
                    if (i + 2 < len(child) and isinstance(child[i+2], nn.ReLU)):
                        group.append(str(i+2))
                        i += 3
                    else:
                        i += 2
                    fusion_groups.append(group)
                else:
                    i += 1
            for group in fusion_groups:
                try:
                    torch.quantization.fuse_modules(child, group, inplace=True)
                    print(f"Fused sequential group {group} in module {name}")
                except Exception as e:
                    print(f"Failed to fuse sequential group {group} in module {name}: {e}")
    return module

# --- SiLU を ReLU に置換する ---
def replace_silu_with_relu(module):
    """
    モジュール内のすべての SiLU (torch.nn.SiLU) を ReLU (torch.nn.ReLU) に置換する再帰的な関数。
    """
    for name, child in module.named_children():
        if isinstance(child, nn.SiLU):
            setattr(module, name, nn.ReLU(inplace=True))
            print(f"Replaced SiLU with ReLU in module: {name}")
        else:
            replace_silu_with_relu(child)
    return module

# YOLOv12n モデルを読み込み
model = YOLO('yolov12n.pt')
# 内部の PyTorch モデルに対して融合と置換を実施
# model.model = fuse_model(model.model)
model.model = replace_silu_with_relu(model.model)

results = model.train(
  data='coco.yaml',
  epochs=50, 
  batch=48, 
  imgsz=640,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  workers= 6,
  device="0",
)

# 変更後のモデル構造を文字列に変換してファイルに保存（デバッグ用）
model_structure = str(model.model)
with open("modified_model.txt", "w") as f:
    f.write(model_structure)

# 推論を実行する
results = model("./dog-640.jpg")

# 結果画像（bboxが重畳された画像）を取得する
annotated_img = results[0].plot()

# 内部の PyTorch モデル構造を文字列に変換
model_structure = str(model.model)

# 結果を model.txt に保存する
with open("model.txt", "w") as f:
    f.write(model_structure)

print("モデル構造を model.txt に保存した。")

# 結果画像を保存する
cv2.imwrite("dog-640-annotated.jpg", annotated_img)

# 検出したbboxのクラスと座標を出力する
for box in results[0].boxes:
    # bboxの座標 (xyxy) を取得。GPU上のテンソルの場合はCPUへ移動しnumpy配列に変換
    xyxy = box.xyxy.cpu().numpy() if box.xyxy.is_cuda else box.xyxy.numpy()
    # クラス番号を整数に変換
    cls = int(box.cls)
    print(f"クラス: {cls}, 座標: {xyxy}")

#evaluate model
model.val(data='coco.yaml', save_json=True)
print("model evaluation done.")