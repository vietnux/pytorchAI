import os
import utils.generate as genImg
import utils.img2Img as img2Img
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import filedialog
from PIL import Image, ImageTk

print("Ứng dụng đang chạy... Mở cửa sổ giao diện!")
file_selected = ""


def view_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((400, 400))
    img = ImageTk.PhotoImage(img)

    def update_image():
        canvas_img.create_image(0, 0, anchor="nw", image=img)
        canvas_img.image = img
        canvas_img.config(scrollregion=canvas.bbox("all"))
        status_label.config(text="Tạo ảnh thành công!")

    root.after(10, update_image)


def generate_image():
    global file_selected
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

    status_label.config(text="Khởi động mô hình")
    root.update_idletasks()
    if file_selected:
        img2Img.init_model(combo.get(), status_label)
        img2Img.generate_image(
            file_selected,
            prompt,
            width,
            height,
            NUM_STEPS,
            status_label,
            progress_bar,
            callback=view_image,
            root=root,
        )
    else:
        genImg.init_model(combo.get(), status_label)
        genImg.generate_image(
            prompt,
            width,
            height,
            NUM_STEPS,
            status_label,
            progress_bar,
            callback=view_image,
            root=root,
        )


def list_models():
    directory = os.path.join(os.path.dirname(__file__), "models")
    return [entry.name for entry in os.scandir(directory) if entry.is_dir()]


def browse_folder():
    """Mở hộp thoại để người dùng chọn thư mục, sau đó cập nhật Combobox."""
    #
    # if folder_selected:
    subfolders = list_models()
    combo["values"] = subfolders
    if subfolders:
        combo.current(0)  # Chọn mục đầu tiên
    else:
        combo.set("")  # Xóa lựa chọn nếu không có thư mục con


def choose_image():
    global file_selected
    file_selected = filedialog.askopenfilename(
        title="Chọn tệp ảnh",
        filetypes=[
            ("Tệp ảnh", "*.png *.jpg *.jpeg *.gif *.bmp *.ico"),
            ("Tất cả các tệp", "*.*"),
        ],
    )
    print("File ảnh: " + file_selected)
    if file_selected:
        display_image(file_selected)

def display_image(file_path):
    image = Image.open(file_path)
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

# Xóa ảnh đầu vào
def clear_image():
    global file_selected
    file_selected = ""
    image_label.config(image='')
    image_label.image = None


# Tạo cửa sổ giao diện
root = tk.Tk()
root.title("AI Image Generator")
root.geometry("1024x600")
# root.configure(bg="#f0f0f0")

# canvas = tk.Canvas(root)
# Chia giao diện làm 2 phần ngang bằng nhau
main_frame = tk.Frame(root)
main_frame.pack(fill="both", expand=True)

left_frame = tk.Frame(main_frame)
left_frame.pack(side="left", fill="both", expand=True)

right_frame = tk.Frame(main_frame, width=512)
right_frame.pack(side="right", fill="both", expand=True)

canvas = tk.Canvas(left_frame)
# Tạo khung cuộn
scroll_y = tk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
scroll_x = tk.Scrollbar(left_frame, orient="horizontal", command=canvas.xview)

canvas.config(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
scroll_y.pack(side="right", fill="y")
scroll_x.pack(side="bottom", fill="x")
canvas.pack(expand=True, fill="both")

frame = tk.Frame(canvas, padx=20, pady=20, relief="groove", bd=2)
frame.pack(fill="x", expand=True)  # frame.pack(fill="both", expand=True)
window_window = canvas.create_window((0, 0), window=frame, anchor="nw", width=canvas.winfo_width() )

def on_canvas_resize(event):
    # Cập nhật lại chiều rộng của frame khi canvas thay đổi kích thước
    canvas.itemconfig(window_window, width=event.width)
    # Cập nhật kích thước khung mỗi khi thay đổi kích thước canvas
canvas.bind("<Configure>", on_canvas_resize)

# Cập nhật vùng cuộn khi nội dung thay đổi
def update_scroll_region(event=None):
    canvas.config(scrollregion=canvas.bbox("all"))


frame.bind("<Configure>", update_scroll_region)

# Tiêu đề
tk.Label(frame, text="AI Image Generator", font=("Arial", 16, "bold")).pack(
    pady=5, fill="x", expand=True
)

# Nút để chọn thư mục
browse_button = ttk.Button(frame, text="Choose image", command=choose_image)
browse_button.pack(pady=10)



tk.Label(frame, text="Nhập mô tả hình ảnh:", font=("Arial", 12)).pack(pady=10)
prompt_entry = tk.Text(frame, width=50, height=4, font=("Arial", 12), wrap="word")
prompt_entry.pack(pady=5)

# Nhập kích thước ảnh
size_label_frame = tk.Frame(frame)
size_label_frame.pack(pady=5, fill="x", expand=True)
tk.Label(size_label_frame, text="Width", font=("Arial", 12)).pack(
    side="left", pady=10, fill="x", expand=True
)
tk.Label(size_label_frame, text="Height", font=("Arial", 12)).pack(
    side="left", pady=10, fill="x", expand=True
)
tk.Label(size_label_frame, text="Number step (1 - 100):", font=("Arial", 12)).pack(
    side="left", pady=5, fill="x", expand=True
)

size_frame = tk.Frame(frame)
size_frame.pack(pady=5, fill="x", expand=True)

prompt_width = tk.Entry(size_frame, width=10)
prompt_width.insert(0, "512")  # Giá trị mặc định
prompt_width.pack(side="left", padx=5, fill="x", expand=True)



prompt_height = tk.Entry(size_frame, width=10)
prompt_height.insert(0, "512")  # Giá trị mặc định
prompt_height.pack(side="left", padx=5, fill="x", expand=True)


step_scale = tk.Scale(size_frame, from_=1, to=100, orient="horizontal")
step_scale.set(30)  # Giá trị mặc định: Nếu preview nhanh → dùng 20–30; Nếu xuất ảnh chính thức → dùng 40–50+
step_scale.pack(side="left", padx=5, fill="x", expand=True)


prompt_suggestions = {
    "Anime": "anime style, vibrant colors, sharp lines, fantasy background, high detail",
    "Fantasy": "epic landscape, magical creatures, surreal lighting, ancient ruins, detailed",
    "Realistic": "ultra realistic photo, 8k, shallow depth of field, natural lighting",
    "Cyberpunk": "cyberpunk cityscape, neon lights, futuristic buildings, dark tone",
    "Cute Animal": "adorable kitten, big eyes, soft lighting, cartoon style, HD",
    "Sci-fi": "space station, futuristic design, robots, clean lines, glowing elements",

    "European Fairy Tale": "Watercolor fantasy illustration style, bright and dreamy atmosphere, high-detail character design, elegant fairy tale costume, luminous lighting, soft pastel colors, magical realism, inspired by classic European storybooks, cinematic composition",    
    "Vietnamese Fairy Tale": "Vietnamese folk tale illustration style, watercolor-inspired, delicate brush textures, bright and poetic atmosphere, traditional áo dài or ancient costumes, tropical nature elements, rice fields, bamboo, mythical animals, misty lighting, culturally rich ornamentation, inspired by Đông Hồ folk art and Vietnamese temple murals.",    
    "Japanese Fairy Tale": "a Japanese fairy tale illustration featuring a brave boy with a wooden sword, cherry blossom trees, floating lanterns, a Koi fish spirit companion, traditional kimono, watercolor art style, bright colors, Studio Ghibli-inspired, clean and modern fairy tale mood",
    
    "Chinese Fairy Tale": "a fantasy Chinese fairy tale painting of a dragon flying over lotus lakes, a princess in flowing hanfu on a golden bridge, misty mountains, watercolor brush strokes, vibrant but soft color palette, magical realism, traditional Chinese folklore, elegant and dreamlike",
}

def add_prompt_suggestion():
    selected = suggestion_combo.get()
    if selected in prompt_suggestions:
        prompt = prompt_suggestions[selected]
        current_text = prompt_entry.get("1.0", tk.END).strip()
        if current_text:
            prompt += ", " + current_text
        prompt_entry.delete("1.0", tk.END)
        prompt_entry.insert(tk.END, prompt)

suggestion_combo = ttk.Combobox(frame, values=list(prompt_suggestions.keys()), state="readonly")
suggestion_combo.set("Chọn chủ đề gợi ý")  # Placeholder
suggestion_combo.pack(pady=5, fill="x")

add_suggestion_button = ttk.Button(frame, text="Thêm gợi ý", command=add_prompt_suggestion)
add_suggestion_button.pack(pady=5)

def open_prompt_popup():
    popup = tk.Toplevel(root)
    popup.title("Thư viện prompt nâng cao")
    popup.geometry("400x400")

    prompts = [
        "Masterpiece, best quality, 8k, ultra detail, stunning light",
        "Fantasy landscape, beautiful scenery, lush forests, glowing effects",
        "Cinematic lighting, portrait, shallow DOF, vibrant color",
        "Cute anime girl, big eyes, colorful hair, school uniform",
        "Cyberpunk street, neon lights, rain, night scene, reflections",
        # 🇪🇺 Phong cách Cổ tích châu Âu (Grimm, Andersen...)
        "a fairy tale illustration of a magical forest with a little girl in red hood, a big-eyed friendly wolf, dreamy castle in the background, watercolor style, pastel colors, bright and airy lighting, european fairy tale, children storybook illustration, fantasy art",
        # 🇻🇳 Phong cách Cổ tích Việt Nam
        "a fairy tale scene of a young Vietnamese girl riding a white buffalo through lush green rice fields, ancient temples in the background, traditional áo dài dress, bamboo forest, watercolor painting, modern and dreamy, clear sky, Vietnamese folklore, serene and magical atmosphere",
        # 🇯🇵 Phong cách Cổ tích Nhật Bản
        "a Japanese fairy tale illustration featuring a brave boy with a wooden sword, cherry blossom trees, floating lanterns, a Koi fish spirit companion, traditional kimono, watercolor art style, bright colors, Studio Ghibli-inspired, clean and modern fairy tale mood",
        # 🇨🇳 Phong cách Cổ tích Trung Hoa
        "a fantasy Chinese fairy tale painting of a dragon flying over lotus lakes, a princess in flowing hanfu on a golden bridge, misty mountains, watercolor brush strokes, vibrant but soft color palette, magical realism, traditional Chinese folklore, elegant and dreamlike",
    ]

    def on_select_prompt(event):
        selected_prompt = prompt_listbox.get(prompt_listbox.curselection())
        prompt_entry.delete("1.0", tk.END)
        prompt_entry.insert(tk.END, selected_prompt)
        popup.destroy()

    prompt_listbox = tk.Listbox(popup, font=("Arial", 11))
    for item in prompts:
        prompt_listbox.insert(tk.END, item)
    prompt_listbox.pack(expand=True, fill="both", padx=10, pady=10)
    prompt_listbox.bind("<Double-Button-1>", on_select_prompt)

popup_prompt_button = ttk.Button(frame, text="Mở thư viện prompt nâng cao", command=open_prompt_popup)
popup_prompt_button.pack(pady=5)

# subfolders = list_models()

# Combobox để hiển thị các thư mục con
combo = ttk.Combobox(frame, state="readonly")
combo.pack(pady=10)
combo.pack(fill="x", expand=True, padx=5)
# combo.bind("<<ComboboxSelected>>", lambda event: genImg.init_model( combo.get() ) )
browse_folder()

generate_button = tk.Button(frame, text="Tạo ảnh", command=generate_image)
generate_button.pack(pady=10)

# clear_button = ttk.Button(frame, text="Xóa ảnh đầu vào", command=clear_image)
# clear_button.pack()


# Nhãn để hiển thị ảnh
# image_label = tk.Label(root)
image_label = tk.Label(right_frame)
image_label.pack(pady=10)
# Thanh tiến trình
progress_bar = ttk.Progressbar(
    right_frame, orient="horizontal", length=300, mode="determinate"
)
progress_bar.pack(pady=5)

# Nhãn trạng thái
status_label = tk.Label(right_frame, text="", font=("Arial", 10), fg="red")
status_label.pack(pady=5)

# Khung chứa ảnh có thanh cuộn
image_frame = tk.Frame(right_frame, bg="white", padx=20, pady=20, relief="ridge", bd=2)
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
