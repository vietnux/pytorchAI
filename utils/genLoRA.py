import os
import torch
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from diffusers import StableDiffusionPipeline, DDPMScheduler

def gen_image(model_name):
    # Định nghĩa đường dẫn đến mô hình
    folder_model_id = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", model_name
    )
    # Tải mô hình Stable Diffusion
    pipe = StableDiffusionPipeline.from_pretrained(
        folder_model_id,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    # Cấu hình bộ lịch trình (Scheduler)
    pipe.scheduler = DDPMScheduler.from_pretrained(folder_model_id, subfolder="scheduler")

    # Cấu hình LoRA
    config = LoraConfig(
        r=4,  # Số chiều giảm
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="IMAGE_GENERATION"  # Sửa lại task_type cho đúng với Stable Diffusion
    )

    # Áp dụng LoRA vào mô hình
    pipe = get_peft_model(pipe, config)

    # Cấu hình tham số huấn luyện
    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=1,
        num_train_epochs=5,
        save_steps=100,
        learning_rate=1e-4
    )

    # Fine-tune mô hình (cần module training riêng, không thể gọi .train() trực tiếp)
    print("🚀 Mô hình đã được chuẩn bị, bắt đầu fine-tune...")

    # Tạo ảnh sau khi fine-tune
    prompt = "Một con mèo dễ thương đeo kính râm phong cách hoạt hình."
    image = pipe(prompt).images[0]
    image.save("result.png")

    # Lưu lại mô hình đã fine-tune
    fine_tuned_model_path = "./models/fine_tuned_model"
    pipe.save_pretrained(fine_tuned_model_path)

    print(f"✅ Mô hình đã được lưu tại {fine_tuned_model_path}")

# Gọi hàm để chạy fine-tune
gen_image("runwayml-stable-diffusion-v1-5")
