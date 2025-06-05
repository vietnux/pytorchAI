# 2. Chỉnh sửa hình ảnh với Img2Img
# Phương pháp Img2Img cho phép bạn biến đổi toàn bộ hình ảnh dựa trên một mô tả văn bản, 
# hữu ích khi bạn muốn thay đổi phong cách hoặc nội dung tổng thể của hình ảnh.​
# Yêu cầu:
#     Cài đặt thư viện diffusers và transformers.​
#     Tải mô hình Stable Diffusion phù hợp, ví dụ: CompVis/stable-diffusion-v1-4.
import os
import torch
from PIL import Image, ImageTk
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from diffusers.callbacks import SDXLCFGCutoffCallback
from unidecode import unidecode
from tkinter import messagebox
from transformers import CLIPTokenizer, CLIPTextModel
from safetensors.torch import load_file

import threading
import re
import gc
import math

# Kiểm tra GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(torch.cuda.memory_allocated() / 1024**3, "GB")
generator = torch.Generator(device="cpu").manual_seed(2628670641)
# Số bước suy luận mặc định
NUM_STEPS = 100
pipe = None
model_id = ''

def init_model( model_name, status_label ) :
    global pipe, model_id
    # Load mô hình Stable Diffusion
    print("Đang tải mô hình Stable Diffusion...")
    
    # Load mô hình từ thư mục cục bộ trong venv
    # model_id = os.path.join( os.path.dirname(os.path.dirname(__file__)), "models", "Linaqruf--animagine-xl")
    # folder_model_id = os.path.join( os.path.dirname(os.path.dirname(__file__)), "models", 'CompVis-stable-diffusion-v1-4')
    folder_model_id = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "models", model_name
    )
    if( folder_model_id != model_id and pipe == None ) :
        model_id = folder_model_id
        # model_id = "CompVis/stable-diffusion-v1-4" # Linaqruf--animagine-xl
        print("Path..."+model_id )
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16", #chỉ dùng khi tải mô hình từ Hugging Face Hub và có phiên bản fp16.
            use_safetensors=True,  # Sử dụng .safetensors nếu không có .bin
            # added_cond_kwargs={}
            safety_checker=None
        )

        pipe.to(device)
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    
    # https://civitai.com/
    # Load LoRA từ file .safetensors
    # lora_path = os.path.join( os.path.dirname(os.path.dirname(__file__)), "models/erza_ix-000008.safetensors") # "./models/erza_ix-000008.safetensors"
    lora_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models/Fairy_Princess.safetensors",
    )

    # status_label.config(text="Đang tải LoRA...")
    # if not os.path.exists(lora_path):
    #     print("⚠️ Không tìm thấy file LoRA:", lora_path)
    # else:
    #     load_lora(pipe, lora_path)

    pipe.to(device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )

    print("Mô hình đã sẵn sàng!")

def load_lora(pipe, lora_path):
    try:
        print("Available modules in UNet:")
        # Tải nội dung file
        tensors = load_file(lora_path)

        # In ra danh sách tensor và kích thước
        # for name, tensor in tensors.items():
        #     print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")

        if any(dim == 2048 for t in tensors.values() for dim in t.shape):
            print("✅ Có vẻ là LoRA cho SDXL")
        else:
            print("⚠️ Không phải SDXL, có thể là SD1.5 hoặc LoRA không tương thích")

        # for name, _ in pipe.unet.named_modules():
        #     print(name)


        pipe.load_lora_weights(lora_path, adapter_name="my_lora")
        pipe.set_adapters(["my_lora"])

        pipe.unload_lora_weights()
        torch.cuda.empty_cache()
        gc.collect()
        # with torch.no_grad():
        #     out_after = pipe.unet(dummy_input, return_dict=False)
        print("✅ LoRA loaded thành công!")
    except ValueError as e:
        print("⚠️ Không thể load LoRA, có thể không tương thích với SDXL.")
        print(e)
        # messagebox.showerror("Lỗi LoRA", f"Không thể load LoRA tương ứng:\n{str(e)}")


def preprocess_text(text):
    """Chuyển tiếng Việt thành không dấu để mô hình dễ hiểu hơn"""
    return unidecode(text)

# Hàm tạo ảnh từ văn bản
def generate_image( file_selected, prompt, width, height, NUM_STEPS, status_label, progress_bar, callback, root ):
    global pipe
    # a beautiful girl with big eye, skin, and long hair, t-shirt, bursting with vivid color.
    # Xử lý tiếng Việt trước khi đưa vào mô hình
    processed_prompt = preprocess_text(prompt)
    processed_prompt, extra_prompt = split_prompt_sdxl(
        processed_prompt
    )  # 

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
        root.after(10, update_status, progress)
        return callback_kwargs

    negative_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    latent = torch.randn((1, 4, height // 8, width // 8), device=device)

    # Tải hình ảnh gốc và mặt nạ
    init_image = Image.open(file_selected).convert("RGB")

    def run_pipeline():
        torch.cuda.empty_cache()
        try:
            # Tham số strength trong Img2Img xác định mức độ thay đổi so với hình ảnh gốc. 
            # Giá trị gần 1.0 sẽ tạo ra nhiều thay đổi hơn, trong khi giá trị thấp hơn sẽ giữ lại 
            # nhiều đặc điểm của hình ảnh ban đầu hơn.
            image = pipe(
                prompt=processed_prompt,
                prompt_2=extra_prompt,
                negative_prompt=negative_prompt,
                height=math.floor(height / 8) * 8,
                width=math.floor(width / 8) * 8,
                image=init_image, strength=0.75,
                target_size=(1024,1024),
                original_size=(4096,4096),
                guidance_scale=6.5,
                num_inference_steps=NUM_STEPS, 
                generator=generator,
                callback_on_step_end=callback_dynamic_cfg,
                # callback_on_step_end_tensor_inputs=['prompt_embeds'],
                latent=latent
            ).images[0]

            status_label.config(text=f"Đang lưu ảnh... !")

            # Tạo thư mục 'output' nếu chưa có
            os.makedirs("output", exist_ok=True)
            image_path = os.path.join(os.getcwd(),"output",  sanitize_filename(processed_prompt)+".png")
            image.save(image_path)
            callback(image_path)

        except Exception as e:
            error_message = f"Có lỗi xảy ra: {e}"
            print(error_message)
            root.after(10, lambda err=error_message: messagebox.showerror("Lỗi", err))
            root.after(10, lambda: status_label.config(text="Lỗi khi tạo ảnh!"))

    # Chạy trên luồng mới để không làm treo giao diện
    threading.Thread(target=run_pipeline, daemon=True).start()

# Làm sạch prompt để dùng làm tên file
def sanitize_filename(prompt):
    # Chuyển về chữ thường, thay thế khoảng trắng bằng gạch dưới và bỏ ký tự không hợp lệ
    return re.sub(r'[^\w\-_.]', '_', prompt.strip().lower())

def split_prompt_sdxl(prompt, tokenizer=None):
    """
    Tách prompt đầu vào thành tuple (main_prompt, extra_prompt) để phù hợp với giới hạn token của SDXL.
    Mỗi phần được giới hạn ở mức tối đa 77 tokens.
    """
    if tokenizer is None:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # Tách dựa trên dấu phân cách ngữ nghĩa (ưu tiên)
    parts = re.split(r'[.;,!?]\s*', prompt)
    processed_prompt = parts[0].strip()
    extra_prompt = ' '.join(parts[1:]).strip() if len(parts) > 1 else ""

    # Đảm bảo không vượt token
    processed_tokens = tokenizer.tokenize(processed_prompt)
    extra_tokens = tokenizer.tokenize(extra_prompt)
    # tokens = tokenizer.tokenize(prompt)
    max_token = 77

    # if len(tokens) <= max_token:
    #     return prompt, ""

    # first_tokens = tokens[:max_token]
    # second_tokens = tokens[max_token : max_token * 2]

    # main_prompt = tokenizer.convert_tokens_to_string(first_tokens)
    # extra_prompt = tokenizer.convert_tokens_to_string(second_tokens)
    
    processed_prompt = tokenizer.convert_tokens_to_string(processed_tokens[:max_token])
    extra_prompt = tokenizer.convert_tokens_to_string(extra_tokens[:max_token])

    print(f"✅ Prompt chính: {len(processed_tokens)} tokens")
    print(f"✅ Prompt phụ: {len(extra_tokens)} tokens")

    return processed_prompt, extra_prompt
