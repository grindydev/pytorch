from pathlib import Path
import torch
import torch.nn as nn
import torch.quantization
from tqdm import tqdm

import helper_utils

# ====================== DEVICE SETUP ======================
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
# elif torch.backends.mps.is_available():
#     DEVICE = torch.device('mps')
#     print("🚀 Using MPS — Apple Silicon GPU acceleration!")
else:
    DEVICE = torch.device('cpu')
print(f"Using Device: {DEVICE}")


# ====================== MODEL DEFINITION ======================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        x = x.view(-1, 512 * 2 * 2)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# ====================== LOAD DATA ======================
data_path = Path.cwd() / 'data/CIFAR10_data'
trainloader, testloader = helper_utils.load_cifar10(data_path=data_path)


# ====================== LOAD BASELINE MODEL ======================
print('\n' + '='*30 + " Loading Baseline Model " + '='*30 + '\n')

baseline_model_path = Path.cwd() / 'data/baseline_pretrained_model/cifar10_cnn_30_epochs_best.pt'

baseline_model = CNN()
baseline_model.load_state_dict(torch.load(baseline_model_path, map_location=DEVICE))
baseline_model = baseline_model.to(DEVICE)
baseline_model.eval()

baseline_model_size = helper_utils.get_model_size(baseline_model)
baseline_model_inf_time = helper_utils.measure_average_inference_time_ms(baseline_model)

print(f"Baseline model size: {baseline_model_size:.2f} MB")
print(f"Baseline model inference time: {baseline_model_inf_time:.2f} ms")


# ====================== DYNAMIC QUANTIZATION ======================
print('\n' + '='*30 + " Dynamic Quantization " + '='*30 + '\n')

print("--- Weight dtypes before quantization ---")
for name, module in baseline_model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        print(f"Layer: {name:<10} | Weight dtype: {module.weight.dtype}")

torch.backends.quantized.engine = 'qnnpack'
print("✅ Quantized engine set to 'qnnpack'")

baseline_model_cpu = baseline_model.to('cpu')

quantized_dynamic_model = torch.quantization.quantize_dynamic(
    baseline_model_cpu, {nn.Linear}, dtype=torch.qint8
)

model_path = Path.cwd() / 'data/models'
model_path.mkdir(parents=True, exist_ok=True)
torch.save(quantized_dynamic_model.state_dict(), model_path / 'cifar10_cnn_quantized_dynamic.pth')

print("\n--- Weight dtypes after dynamic quantization ---")
for name, module in quantized_dynamic_model.named_modules():
    if isinstance(module, torch.nn.quantized.dynamic.Linear):
        print(f"Layer: {name:<10} | Weight dtype: {module.weight().dtype}")

print("\n✅ Dynamic quantization completed and model saved!")

quantized_dynamic_model_size = helper_utils.get_model_size(quantized_dynamic_model)
quantized_dynamic_model_inf_time = helper_utils.measure_average_inference_time_ms(quantized_dynamic_model)

helper_utils.print_terminal_comparison_table(
    baseline_model_size=baseline_model_size,
    baseline_model_time=baseline_model_inf_time,
    quantized_model_size=quantized_dynamic_model_size,
    quantized_model_time=quantized_dynamic_model_inf_time,
    quantization_type="Dynamic"
)


# ====================== STATIC QUANTIZATION ======================
print('\n' + '='*30 + " Static Quantization " + '='*30 + '\n')

class QuantizedCNN(nn.Module):
    def __init__(self):
        super(QuantizedCNN, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.reshape(-1, 512 * 2 * 2)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x


quantized_static_model = QuantizedCNN()
quantized_static_model.load_state_dict(baseline_model.state_dict())
quantized_static_model.eval()

quantized_static_model.qconfig = torch.quantization.get_default_qconfig('x86')
torch.quantization.prepare(quantized_static_model, inplace=True)

print("--- Weight dtypes before static conversion ---")
for name, module in quantized_static_model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        print(f"Layer: {name:<10} | Weight dtype: {module.weight.dtype}")


def calibrate(model, data_loader, num_batches=50):
    model.eval()
    with torch.no_grad():
        progress_bar = tqdm(data_loader, total=num_batches, desc="Calibrating")
        for i, (image, _) in enumerate(progress_bar):
            if i >= num_batches:
                break
            model(image)
    print(f"\nCalibration finished after processing {num_batches} batches.")


calibrate(quantized_static_model, testloader)
torch.quantization.convert(quantized_static_model, inplace=True)

print("\n--- Weight dtypes after static conversion ---")
for name, module in quantized_static_model.named_modules():
    if isinstance(module, (torch.nn.quantized.Conv2d, torch.nn.quantized.Linear)):
        print(f"Layer: {name:<10} | Weight dtype: {module.weight().dtype}")

torch.save(quantized_static_model.state_dict(), model_path / 'cifar10_cnn_quantized_static.pth')

quant_static_model_size = helper_utils.get_model_size(quantized_static_model)
quant_static_model_inf_time = helper_utils.measure_average_inference_time_ms(quantized_static_model)

helper_utils.print_terminal_comparison_table(
    baseline_model_size=baseline_model_size,
    baseline_model_time=baseline_model_inf_time,
    quantized_model_size=quant_static_model_size,
    quantized_model_time=quant_static_model_inf_time,
    quantization_type="Static"
)


# ====================== QUANTIZATION-AWARE TRAINING (QAT) ======================
print('\n' + '='*30 + " Quantization-Aware Training (QAT) " + '='*30 + '\n')

class QATCNN(nn.Module):
    def __init__(self):
        super(QATCNN, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.relu_fc1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 512)
        self.relu_fc2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.bn1(self.conv1(x))); x = self.pool(x)
        x = self.relu2(self.bn2(self.conv2(x))); x = self.pool(x)
        x = self.relu3(self.bn3(self.conv3(x))); x = self.pool(x)
        x = self.relu4(self.bn4(self.conv4(x))); x = self.pool(x)

        x = x.reshape(-1, 512 * 2 * 2)
        x = self.relu_fc1(self.fc1(x))
        x = self.dropout(x)
        x = self.relu_fc2(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self, ['conv1', 'bn1', 'relu1'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv2', 'bn2', 'relu2'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv3', 'bn3', 'relu3'], inplace=True)
        torch.quantization.fuse_modules(self, ['conv4', 'bn4', 'relu4'], inplace=True)
        torch.quantization.fuse_modules(self, ['fc1', 'relu_fc1'], inplace=True)
        torch.quantization.fuse_modules(self, ['fc2', 'relu_fc2'], inplace=True)


qat_model = QATCNN()
qat_model.load_state_dict(baseline_model.state_dict(), strict=False)
qat_model.eval()
qat_model.fuse_model()
qat_model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
qat_model.train()

qat_model = torch.quantization.prepare_qat(qat_model)
qat_model = helper_utils.train_qat(qat_model, trainloader, DEVICE, epochs=5)

final_quantized_qat_model = torch.quantization.convert(qat_model.eval(), inplace=False)

print("\n--- Weight dtypes after final QAT conversion ---")
for name, module in final_quantized_qat_model.named_modules():
    if isinstance(module, (torch.nn.quantized.Conv2d, torch.nn.quantized.Linear)):
        print(f"Layer: {name:<10} | Weight dtype: {module.weight().dtype}")

torch.save(final_quantized_qat_model.state_dict(), model_path / 'cifar10_cnn_qat_quantized.pt')

helper_utils.evaluate_qat(final_quantized_qat_model, testloader)

qat_model_size = helper_utils.get_model_size(final_quantized_qat_model)
qat_model_inf_time = helper_utils.measure_average_inference_time_ms(final_quantized_qat_model)

helper_utils.print_terminal_comparison_table(
    baseline_model_size=baseline_model_size,
    baseline_model_time=baseline_model_inf_time,
    quantized_model_size=qat_model_size,
    quantized_model_time=qat_model_inf_time,
    quantization_type="QAT"
)


# ====================== FINAL SUMMARY & LEARNING CONTENT ======================
print('\n' + '='*80)
print("                  FINAL COMPARISON - ALL MODELS")
print('='*80)

all_stats = {
    "Baseline": (baseline_model_size, baseline_model_inf_time),
    "Dynamic": (quantized_dynamic_model_size, quantized_dynamic_model_inf_time),
    "Static": (quant_static_model_size, quant_static_model_inf_time),
    "QAT": (qat_model_size, qat_model_inf_time)
}

print(f"| {'Model Type':<12} | {'Size (MB)':>10} | {'Inference (ms)':>14} |")
print("-"*70)
for name, (size, time) in all_stats.items():
    print(f"| {name:<12} | {size:>10.2f} | {time:>14.2f} |")
print('='*80)

print("\n🎓 LEARNING SUMMARY - Advantages & When to Use Each Quantization Method\n")

print("""**Dynamic Quantization**
• Best Practices: Simplest method. No model changes or calibration needed.
• Use Cases: Great starting point. Best when you cannot retrain the model or provide calibration data.
  Useful for models with highly variable activations (e.g. LSTMs, Transformers).
  Good for quick size reduction with moderate speed-up.

**Static Quantization**
• Best Practices: Requires calibration on representative data + layer fusion (Conv-BN-ReLU).
• Use Cases: Excellent for CNNs on CPUs. Gives better compression and speed than Dynamic.
  Ideal for server-side deployment where you can run calibration once.

**Quantization-Aware Training (QAT)**
• Best Practices: Requires model fusion + short fine-tuning (1-5 epochs).
• Use Cases: Highest accuracy among quantized models. Use when you need maximum performance
  and cannot accept accuracy drop from post-training quantization.
  Best for production models where accuracy is critical.""")

print("\n✅ All models saved in data/models/ folder.")
print("You have now mastered the full quantization pipeline!")