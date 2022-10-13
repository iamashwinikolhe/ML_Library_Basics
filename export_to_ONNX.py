import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)  # download pretrained model
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
input_names = ["input"]
output_names = ["output"]

torch.onnx.export(model,
                  dummy_input,
                  "my_resnet50_onnx.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  export_params=True,
                  )                                 # convert the model to ONNX
print("Model has been exported to ONNX format")
