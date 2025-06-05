# 1. Inpainting với Stable Diffusion
# Inpainting cho phép bạn chỉnh sửa một phần cụ thể của hình ảnh bằng cách 
# cung cấp một mặt nạ (mask) 
# xác định khu vực cần thay đổi và một mô tả văn bản về nội dung mong muốn.​
# Yêu cầu:
#     Cài đặt thư viện diffusers và transformers:
import os
import torch
from PIL import Image, ImageTk
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.callbacks import SDXLCFGCutoffCallback
from unidecode import unidecode
from tkinter import messagebox

import threading

# Kiểm tra GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(torch.cuda.memory_allocated() / 1024**3, "GB")
generator = torch.Generator(device="cpu").manual_seed(2628670641)
# Số bước suy luận mặc định
NUM_STEPS = 100
pipe = None
model_id = ''

def init_model( model_name ) :
    global pipe, model_id
    # Load mô hình Stable Diffusion
    print("Đang tải mô hình Stable Diffusion...")
    
    # Load mô hình từ thư mục cục bộ trong venv

    # Tạo pipeline và load mô hình cục bộ trong venv
    # model_id = os.path.join( os.path.dirname(os.path.dirname(__file__)), "models/sd3.5_large_turbo.safetensors")
    # pipe = StableDiffusionPipeline.from_single_file(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    #     variant="fp16",
    #     low_cpu_mem_usage=True  # Giảm tiêu tốn RAM
    # )

    # model_id = os.path.join( os.path.dirname(os.path.dirname(__file__)), "models", "Linaqruf--animagine-xl")
    folder_model_id = os.path.join( os.path.dirname(os.path.dirname(__file__)), "models", model_name)
    if( folder_model_id != model_id and pipe == None ) :
        model_id = folder_model_id
        # model_id = "runwayml/stable-diffusion-v1-5"
        # model_id = "stabilityai/stable-diffusion-xl-base-1.0" #stabilityai/stable-diffusion-xl-base-1.0
        model_id = "stabilityai/stable-diffusion-2-inpainting" # Linaqruf--animagine-xl
        
        print("Path..."+model_id )
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16", #chỉ dùng khi tải mô hình từ Hugging Face Hub và có phiên bản fp16.
            # use_safetensors=True  # Sử dụng .safetensors nếu không có .bin
            # added_cond_kwargs={}
        )

        pipe.to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    print("Mô hình đã sẵn sàng!")

def preprocess_text(text):
    """Chuyển tiếng Việt thành không dấu để mô hình dễ hiểu hơn"""
    return unidecode(text)

# Hàm tạo ảnh từ văn bản
def generate_image( file_selected, file_mask, prompt, width, height, NUM_STEPS, status_label, progress_bar, callback, root ):
    global pipe
    # a beautiful girl with big eye, skin, and long hair, t-shirt, bursting with vivid color.
    # Xử lý tiếng Việt trước khi đưa vào mô hình
    processed_prompt = prompt # preprocess_text(prompt)

    # Cập nhật tiến trình lên giao diện
    def update_status(progress):
        status_label.config(text=f"Đang tạo ảnh... {progress}%")
        progress_bar["value"] = progress
        root.update_idletasks()

    def callback_dynamic_cfg(pipe, step_index, timestep, callback_kwargs):
            # adjust the batch_size of prompt_embeds according to guidance_scale
            if step_index == int(pipe.num_timesteps * 0.05):
                    prompt_embeds = callback_kwargs["prompt_embeds"]
                    prompt_embeds = prompt_embeds.chunk(2)[-1]

                    # update guidance_scale and prompt_embeds
                    pipe._guidance_scale = 0.0
                    callback_kwargs["prompt_embeds"] = prompt_embeds
                    
            progress = int((step_index / NUM_STEPS) * 100)
            root.after(3, update_status, progress)
            return callback_kwargs
    

    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    latent = torch.randn((1, 4, height // 8, width // 8), device=device)

    # Tải hình ảnh gốc và mặt nạ
    init_image = Image.open(file_selected).convert("RGB")
    mask_image = Image.open(file_mask).convert("RGB")

    def run_pipeline():
        try:
            image = pipe(
                prompt=processed_prompt,
                negative_prompt=negative_prompt,
                height=height, 
                width=width, 
                image=init_image, mask_image=mask_image,
                target_size=(1024,1024),
                original_size=(4096,4096),
                guidance_scale=6.5,
                num_inference_steps=NUM_STEPS, 
                generator=generator,
                callback_on_step_end=callback_dynamic_cfg,
                callback_on_step_end_tensor_inputs=['prompt_embeds'],
                latent=latent
            ).images[0]

            image_path = "output.png"
            image.save(image_path)
            callback(image_path)

        except Exception as e:
            error_message = f"Có lỗi xảy ra: {e}"
            print(error_message)
            root.after(10, lambda err=error_message: messagebox.showerror("Lỗi", err))
            root.after(10, lambda: status_label.config(text="Lỗi khi tạo ảnh!"))

    # Chạy trên luồng mới để không làm treo giao diện
    threading.Thread(target=run_pipeline, daemon=True).start()
