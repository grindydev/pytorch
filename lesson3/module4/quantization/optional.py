from pathlib import Path
import torch
import torch.nn as nn
import torch.quantization
from IPython.display import Image as DisplayImage

import helper_utils

# ====================== DEVICE SETUP ======================
# LEARNING: BLIP VQA is a large model. We run everything on CPU for stability.
DEVICE = torch.device('cpu')
print(f"Using Device: {DEVICE}")

# ====================== LOAD BLIP VQA MODEL ======================
local_path = Path.cwd() / 'data/blip-vqa-base-local'
blip_vqa_model, blip_vqa_processor = helper_utils.get_blip_vqa_model_and_processor(local_path=local_path)

blip_vqa_model.to(DEVICE)
blip_vqa_model.eval()

blip_baseline_model_size = helper_utils.get_model_size(blip_vqa_model)

print(f"Baseline BLIP VQA Model Size: {blip_baseline_model_size:.2f} MB")

# ====================== IMAGE UPLOAD & DISPLAY ======================
output_image_folder = Path.cwd() / 'images'
helper_utils.upload_jpg_widget(output_image_folder=output_image_folder)

image_path = Path.cwd() / 'data/images/eiffel_tower.jpg'   # default example

# Display the image
DisplayImage(filename=image_path, width=400, height=400)

question = "Describe the scene in the image."

baseline_answer, blip_model_inf_time = helper_utils.perform_vqa(
    blip_vqa_model, blip_vqa_processor, image_path, question
)

print(f"\nQuestion: {question}")
print(f"Baseline Model Answer: {baseline_answer}")
print(f"Baseline Inference Time: {blip_model_inf_time:.4f} s")


# ====================== DYNAMIC QUANTIZATION ======================
print('\n' + '='*50 + " Dynamic Quantization on BLIP VQA " + '='*50 + '\n')

torch.backends.quantized.engine = 'qnnpack'
print("✅ Quantized engine set to 'qnnpack' (required for Mac)")

try:
    # LEARNING: We only quantize Linear layers because BLIP has many of them.
    # Conv layers in BLIP often cause engine errors on Mac.
    quantized_blip_vqa_model = torch.quantization.quantize_dynamic(
        blip_vqa_model.to('cpu'),
        {nn.Linear},
        dtype=torch.qint8
    )
    print("✅ Dynamic quantization applied successfully!")

except Exception as e:
    print(f"❌ Dynamic quantization failed: {e}")
    print("   → This is common on Mac with large models like BLIP.")
    quantized_blip_vqa_model = None   # fallback

# ====================== RESULTS & COMPARISON ======================
if quantized_blip_vqa_model is not None:
    quantized_blip_model_size = helper_utils.get_model_size(quantized_blip_vqa_model)
    
    quantized_answer, quantized_inf_time = helper_utils.perform_vqa(
        quantized_blip_vqa_model, blip_vqa_processor, image_path, question
    )

    print(f"\nQuantized Model Size: {quantized_blip_model_size:.2f} MB")
    print(f"Quantized Inference Time: {quantized_inf_time:.4f} s")
    print(f"Quantized Model Answer: {quantized_answer}")

    # Show nice comparison table
    helper_utils.print_blip_comparison_table(
        question=question,
        baseline_answer=baseline_answer,
        quantized_answer=quantized_answer,
        baseline_size=blip_baseline_model_size,
        quantized_size=quantized_blip_model_size,
        baseline_time_s=blip_model_inf_time,
        quantized_time_s=quantized_inf_time
    )

else:
    print("\n⚠️  Quantization failed, so no comparison table is shown.")
    print("   You can still use the original baseline model.")

print("\n🎉 BLIP VQA quantization script finished!")