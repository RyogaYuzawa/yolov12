from ultralytics import YOLO
import cv2

# model = YOLO('yolov12n.pt')
model = YOLO('best.pt')

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
