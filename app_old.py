import os
import torch
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.callbacks import SDXLCFGCutoffCallback
from unidecode import unidecode

import threading

print("Ứng dụng đang chạy... Mở cửa sổ giao diện!")

# Kiểm tra GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load mô hình Stable Diffusion
print("Đang tải mô hình Stable Diffusion...")
# model_id = "runwayml/stable-diffusion-v1-5"
# Load mô hình từ thư mục cục bộ trong venv
model_id = os.path.join(os.path.dirname(__file__), "pytorch_env/models/models--runwayml--stable-diffusion-v1-5")
# pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32)

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16",
)
pipe.to(device)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

print("Mô hình đã sẵn sàng!")
generator = torch.Generator(device=device).manual_seed(2628670641)

# Số bước suy luận mặc định
NUM_STEPS = 100

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

def preprocess_text(text):
    """Chuyển tiếng Việt thành không dấu để mô hình dễ hiểu hơn"""
    return unidecode(text)

# Hàm tạo ảnh từ văn bản
def generate_image():
    prompt = prompt_entry.get("1.0", tk.END).strip()
    width = prompt_width.get()
    height = prompt_height.get()
    NUM_STEPS = step_scale.get()
    
    if not prompt:
        messagebox.showerror("Lỗi", "Vui lòng nhập mô tả hình ảnh!")
        return
    
    try:
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError("Chiều rộng và chiều cao phải lớn hơn 0!")
    except ValueError:
        messagebox.showerror("Lỗi", "Chiều rộng và chiều cao phải là số nguyên dương!")
        return
    
    # Xử lý tiếng Việt trước khi đưa vào mô hình
    processed_prompt = preprocess_text(prompt)

    status_label.config(text="Đang tạo ảnh... 0%")
    root.update_idletasks()

    def run_pipeline():
        try:
            image = pipe(
                processed_prompt, 
                height=height, 
                width=width,
                negative_prompt="",
                guidance_scale=6.5,
                num_inference_steps=NUM_STEPS, 
                generator=generator,
                callback_on_step_end=callback_dynamic_cfg,
                callback_on_step_end_tensor_inputs=['prompt_embeds']
            ).images[0]

            image_path = "output.png"
            image.save(image_path)

            img = Image.open(image_path)
            img.thumbnail((400, 400))
            img = ImageTk.PhotoImage(img)
            
            def update_image():
                canvas_img.create_image(0, 0, anchor="nw", image=img)
                canvas_img.image = img
                canvas_img.config(scrollregion=canvas.bbox("all"))
                status_label.config(text="Tạo ảnh thành công!")

            root.after(10, update_image)

        except Exception as e:
            error_message = f"Có lỗi xảy ra: {e}"
            print(error_message)
            root.after(10, lambda err=error_message: messagebox.showerror("Lỗi", err))
            root.after(10, lambda: status_label.config(text="Lỗi khi tạo ảnh!"))

    # Chạy trên luồng mới để không làm treo giao diện
    threading.Thread(target=run_pipeline, daemon=True).start()

# Tạo cửa sổ giao diện
root = tk.Tk()
root.title("AI Image Generator")
root.geometry("500x600")
# root.configure(bg="#f0f0f0")

# Tạo khung cuộn
canvas = tk.Canvas(root)
scroll_y = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scroll_x = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)

canvas.config(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
scroll_y.pack(side="right", fill="y")
scroll_x.pack(side="bottom", fill="x")
canvas.pack(expand=True, fill="both")

frame = tk.Frame(canvas, padx=20, pady=20, relief="groove", bd=2)
frame.pack(fill="x", expand=True) #frame.pack(fill="both", expand=True)
canvas.create_window((0, 0), window=frame, anchor="nw")

# Cập nhật vùng cuộn khi nội dung thay đổi
def update_scroll_region(event=None):
    canvas.config(scrollregion=canvas.bbox("all"))

frame.bind("<Configure>", update_scroll_region)

# Tiêu đề
tk.Label(frame, text="AI Image Generator", font=("Arial", 16, "bold")).pack(pady=5,fill="x", expand=True)


tk.Label(frame, text="Nhập mô tả hình ảnh:", font=("Arial", 12)).pack(pady=10)
prompt_entry = tk.Text(frame, width=50, height=4, font=("Arial", 12), wrap="word")
prompt_entry.pack(pady=5)

# Nhập kích thước ảnh
size_label_frame = tk.Frame(frame)
size_label_frame.pack(pady=5,fill="x", expand=True)
tk.Label(size_label_frame , text="Width", font=("Arial", 12)).pack(side="left", pady=10, fill="x", expand=True)
tk.Label(size_label_frame , text="Height", font=("Arial", 12)).pack(side="left", pady=10, fill="x", expand=True)
tk.Label(size_label_frame, text="Number step (1 - 100):", font=("Arial", 12)).pack(side="left", pady=5, fill="x", expand=True)

size_frame = tk.Frame(frame)
size_frame.pack(pady=5,fill="x", expand=True)

prompt_width = tk.Entry(size_frame, width=10)
prompt_width.insert(0, "512")  # Giá trị mặc định
prompt_width.pack(side="left", padx=5,fill="x", expand=True)


prompt_height = tk.Entry(size_frame, width=10)
prompt_height.insert(0, "512")  # Giá trị mặc định
prompt_height.pack(side="left", padx=5,fill="x", expand=True)


step_scale = tk.Scale(size_frame, from_=1, to=100, orient="horizontal")
step_scale.set(50)  # Giá trị mặc định
step_scale.pack(side="left", padx=5,fill="x", expand=True)

generate_button = tk.Button(frame, text="Tạo ảnh", command=generate_image)
generate_button.pack(pady=10)

# Thanh tiến trình
progress_bar = ttk.Progressbar(frame, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=5)

# Nhãn trạng thái
status_label = tk.Label(frame, text="", font=("Arial", 10), fg="red")
status_label.pack(pady=5)

# Khung chứa ảnh có thanh cuộn
image_frame = tk.Frame(frame, bg="white", padx=20, pady=20, relief="ridge", bd=2)
image_frame.pack(expand=True, fill="both", padx=10, pady=10)

tk.Label(image_frame, text="Ảnh đã tạo", font=("Arial", 12, "bold"), bg="white").pack()

# Canvas để hiển thị ảnh có cuộn
canvas_img = tk.Canvas(image_frame, bg="white")
scroll_x = tk.Scrollbar(image_frame, orient="horizontal", command=canvas_img.xview)
scroll_y = tk.Scrollbar(image_frame, orient="vertical", command=canvas_img.yview)

canvas_img.config(xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
scroll_x.pack(side="bottom", fill="x")
scroll_y.pack(side="right", fill="y")
canvas_img.pack(expand=True, fill="both")

# canvas.configure(scrollregion=canvas.bbox("all"))

root.mainloop()
