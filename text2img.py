import torch
from diffusers import StableDiffusionPipeline

# Kiểm tra GPU có sẵn không
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
torch.cuda.empty_cache()
# Load mô hình Stable Diffusion (phiên bản 1.5)
model_id = "runwayml/stable-diffusion-v1-5"


try:
    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe.to(device)
    print("Mô hình đã tải xong!")

    # Nhập mô tả hình ảnh
    prompt = input("Nhập mô tả hình ảnh: ")

    # Tạo ảnh
    image = pipe(prompt).images[0]

    # Lưu ảnh ra file
    image.save("output.png")
    print("Ảnh đã được tạo và lưu dưới tên 'output.png'")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")


