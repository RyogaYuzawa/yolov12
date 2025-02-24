import torch
import torch.nn as nn
from ultralytics import YOLO

# YOLOv12の最終出力（concat前は (1,4,8400) と (1,80,8400) の2つの出力であると仮定）
# このラッパーモジュールは、内部のYOLOモデルの出力を分割して2つのtensorを返す
class YOLOv12NoConcat(nn.Module):
    def __init__(self, yolomodel):
        super(YOLOv12NoConcat, self).__init__()
        self.yolomodel = yolomodel

    def forward(self, x):
        # 通常のforward実行。UltralyticsのYOLOオブジェクトは .model 属性に内部モデルを持つ
        y = self.yolomodel(x)
        # 出力がリストの場合、先頭のtensorを採用する
        if isinstance(y, (list, tuple)):
            y = y[0]
        # ここで、チャネル次元を分割する
        # 前半4チャネルが Mul_2_output、後半80チャネルが Sigmoid_output とする
        mul2_output = y[:, :4, :]
        sigmoid_output = y[:, 4:, :]
        return mul2_output, sigmoid_output

yolo = YOLO('yolov12s.pt')

base_model = yolo.model

# ラッパーモジュールでconcat処理を除去し、2つの出力とする
custom_model = YOLOv12NoConcat(base_model)
custom_model.eval()


dummy_input = torch.randn(1, 3, 640, 640)

# ONNX形式でエクスポートする
# 出力名はそれぞれ "Mul_2_output" と "Sigmoid_output" として別々に指定する
torch.onnx.export(
    custom_model,
    dummy_input,
    "yolov12n_no_concat.onnx",
    input_names=["images"],
    output_names=["Mul_2_output", "Sigmoid_output"],
    dynamic_axes={
        "images": {0: "batch_size"},
        "Mul_2_output": {0: "batch_size"},
        "Sigmoid_output": {0: "batch_size"}
    },
    opset_version=12,
    verbose=True
)
