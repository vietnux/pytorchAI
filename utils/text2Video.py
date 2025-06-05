import os
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import (
    export_to_gif,
    export_to_video,
)  # Hoặc export_to_video nếu bạn muốn file MP4

# ⚠️ Ưu tiên dùng GPU nếu có
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang sử dụng thiết bị: {device}")

# 1. Tải motion adapter của AnimateDiff
local_model_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "models--guoyww--animatediff-motion-adapter-v1-5-2",
)
# adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
adapter = MotionAdapter.from_pretrained(
    local_model_path,
    torch_dtype=(
        torch.float16 if device == "cuda" else torch.float32
    ),  # GPU: float16 | CPU: float32
)
print("ok off!")
# 2. Tải một mô hình text-to-image (ví dụ: Stable Diffusion 1.5)
# Bạn có thể thay thế bằng các checkpoint Stable Diffusion khác mà bạn thích
# model_id = "runwayml/stable-diffusion-v1-5"
model_id = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "runwayml-stable-diffusion-v1-5",
)
print(model_id)
pipe = AnimateDiffPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16" if device == "cuda" else None,
    use_safetensors=True,
)
pipe.to(device)
pipe.scheduler = EulerDiscreteScheduler.from_config(
    pipe.scheduler.config, beta_schedule="linear", clip_sample=False
)

# 3. (Tuỳ chọn) Tối ưu VRAM nếu dùng GPU
if device == "cuda":
    pipe.enable_model_cpu_offload()  # Hỗ trợ tiết kiệm VRAM

# 4. Chuẩn bị prompt
prompt = (
    "A corgi dog wearing a party hat, happily wagging its tail, birthday celebration"
)

# 5. Tạo video frames
print("Bắt đầu tạo animation với AnimateDiff...")
# 2628670641: Gán một giá trị seed cố định — tức là nếu bạn dùng cùng một prompt, cùng mô hình, cùng seed thì kết quả sinh ra (ảnh/video) sẽ giống nhau.
# ❌ Không dùng seed hoặc dùng random.randint(...)	Mỗi lần chạy sẽ ra kết quả khác nhau
generator = torch.Generator(device="cpu").manual_seed(2628670641) # ⚠️ dùng đúng device
output = pipe(
    prompt=prompt,
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=20,
    generator=generator,
)
frames = output.frames[
    0
]  # output.frames là một list chứa các batch frames, ta lấy batch đầu tiên
print("Đã tạo xong các khung hình!")

# 6. Xuất video/gif
# AnimateDiff thường được dùng để tạo các đoạn GIF ngắn, nhưng bạn vẫn có thể xuất ra MP4
# gif_path = "./corgi_party.gif"
# export_to_gif(frames, gif_path, fps=8)
# print(f"Animation đã được lưu tại: {gif_path}")

# Để xuất ra MP4:
# from diffusers.utils import export_to_video
video_path = export_to_video(frames, output_video_path="./corgi_party.mp4", fps=8)
print(f"Video đã được lưu tại: {video_path}")
