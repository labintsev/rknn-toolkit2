from rknn.api import RKNN
import torch
import torch.nn as nn

example_inputs = torch.randn(1, 3, 32, 32)

class OneLayerModel(nn.Module):
    def __init__(self):
        super(OneLayerModel, self).__init__()
        self.flatten = nn.Flatten()
        self.layer = nn.Linear(3072, 10)
        self.layer_0 = nn.Linear(3072, 100)
        self.relu = nn.ReLU()
        self.layer_1 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.flatten(x)
        if torch.rand(1) > 0.5:
            x = self.layer(x)
        else:
            x = self.layer_0(x)
            x = self.relu(x)
            x = self.layer_1(x)
        return x

MODEL_NAME = OneLayerModel.__name__
TORCH_MODEL = f'{MODEL_NAME}.pt'
ONNX_MODEL = f'{MODEL_NAME}.onnx'
RKNN_MODEL = f'{MODEL_NAME}.rknn'
TARGET_PLATFORM = 'rk3588'
DO_RKNN_QUANTIZATION = True


def build_pytorch_model():
    model = OneLayerModel()
    torch.save(model.state_dict(), TORCH_MODEL)


def build_onnx_model():
    model = OneLayerModel()
    model_state = torch.load(TORCH_MODEL)
    model.load_state_dict(model_state)
    model.eval()
    onnx_program = torch.onnx.export(model=model, 
                                     args=(example_inputs), 
                                     f=ONNX_MODEL,
                                     dynamo=True, 
                                     opset_version=17
                                     )
    if onnx_program:
        onnx_program.optimize()
        onnx_program.save(ONNX_MODEL)


if __name__ == '__main__':

    print(f'Build {MODEL_NAME} pytorch and onnx models')
    build_pytorch_model()
    build_onnx_model()
    print('done')

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> config model')
    rknn.config(target_platform=TARGET_PLATFORM)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=ONNX_MODEL)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=DO_RKNN_QUANTIZATION, dataset='./dataset.txt' )
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    outputs = rknn.inference(inputs=[example_inputs.numpy()], data_format='nchw')
    print(outputs)
    print('done')

    rknn.release()
