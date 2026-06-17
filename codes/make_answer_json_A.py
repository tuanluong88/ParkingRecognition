import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# 점유여부 설정(좌클릭(비점유), 스페이스바(점유), q(이전 판별 취소), 우클릭(화면이동), 휠(이미지 확대/축소))

# ==========================================
# 사용자 설정
# ==========================================
CAMERA_ID = "camera_1"  
IMAGE_FOLDER = "/home/kimminsu/바탕화면/second_result_file/second_check/frame_check/camera_1"
DOT_JSON_PATH = "/home/kimminsu/바탕화면/second_result_file/second_check/dot/json_final/dot/parking_spots_zoneA_seg_dot.json"
OUTPUT_GT_JSON = "/home/kimminsu/바탕화면/second_result_file/ground_truth.json"  
#평가하고 싶은 영역 설정
VALID_RANGES = [(11,23),(38, 71), (77, 110), (116, 149), (155, 188)]
DISPLAY_SIZE = 1280  
# ==========================================

class ParkingLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title(f"주차 라벨링 툴 - {CAMERA_ID}")
        
        self.target_spots = self.load_and_filter_spots()
        self.total_dots = len(self.target_spots)
        
        self.current_file_path = None
        self.current_answers = []        
        self.pil_img = None              
        self.zoom_level = 1.0            
        self.pan_x = 0                   
        self.pan_y = 0                   
        self.drag_start_x = 0
        self.drag_start_y = 0

        self.load_master_gt()
        self.setup_ui()

    def load_master_gt(self):
        if os.path.exists(OUTPUT_GT_JSON):
            with open(OUTPUT_GT_JSON, "r", encoding="utf-8") as f:
                try: self.master_gt = json.load(f)
                except: self.master_gt = {}
        else:
            self.master_gt = {}
        if CAMERA_ID not in self.master_gt: self.master_gt[CAMERA_ID] = {}

    def load_and_filter_spots(self):
        with open(DOT_JSON_PATH, "r") as f:
            spots = json.load(f)
        return [s for s in sorted(spots, key=lambda x: int(x["id"])) 
                if any(start <= int(s["id"]) <= end for start, end in VALID_RANGES)]

    def setup_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        tk.Button(top_frame, text="이미지 불러오기", command=self.open_image, font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=5)
        self.lbl_info = tk.Label(top_frame, text="이미지 불러와주세요", fg="blue", font=("Arial", 11))
        self.lbl_info.pack(side=tk.LEFT, padx=10)
        self.lbl_status = tk.Label(top_frame, text="", fg="black", font=("Arial", 11, "bold"))
        self.lbl_status.pack(side=tk.RIGHT, padx=10)

        self.canvas = tk.Canvas(self.root, width=1024, height=1024, bg="gray")
        self.canvas.pack(side=tk.BOTTOM, padx=5, pady=5)
        
        self.canvas.bind("<Button-1>", lambda e: self.record_status(0))
        self.root.bind("<space>", lambda e: self.record_status(1))
        self.root.bind("<q>", lambda e: self.undo_status())
        self.root.bind("<Q>", lambda e: self.undo_status())
        
        self.canvas.bind("<MouseWheel>", self.on_zoom)    
        self.canvas.bind("<Button-4>", self.on_zoom)    
        self.canvas.bind("<Button-5>", self.on_zoom)       
        self.canvas.bind("<Button-3>", self.start_pan)    
        self.canvas.bind("<B3-Motion>", self.drag_pan)    

    def get_filename_key(self, path):
        return os.path.splitext(os.path.basename(path))[0]

    def open_image(self):
        file_path = filedialog.askopenfilename(initialdir=IMAGE_FOLDER)
        if not file_path: return
        self.current_file_path = file_path
        
        key = self.get_filename_key(self.current_file_path)
        
        self.load_master_gt()
        self.current_answers = self.master_gt.get(CAMERA_ID, {}).get(key, [])
        self.pil_img = Image.open(self.current_file_path).resize((DISPLAY_SIZE, DISPLAY_SIZE))
        self.display_image_and_dots()

    def display_image_and_dots(self):
        w, h = int(DISPLAY_SIZE * self.zoom_level), int(DISPLAY_SIZE * self.zoom_level)
        self.tk_img = ImageTk.PhotoImage(self.pil_img.resize((w, h)))
        self.canvas.delete("all")
        x_pos, y_pos = (1024 - w) // 2 + self.pan_x, (1024 - h) // 2 + self.pan_y
        self.canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.tk_img)
        
        for i, ans in enumerate(self.current_answers):
            spot = self.target_spots[i]
            x = int((spot["x"] * (DISPLAY_SIZE / 1024)) * self.zoom_level) + x_pos
            y = int((spot["y"] * (DISPLAY_SIZE / 1024)) * self.zoom_level) + y_pos
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill="#00FF00" if ans == 0 else "#8A2BE2")
            
        if len(self.current_answers) < self.total_dots:
            spot = self.target_spots[len(self.current_answers)]
            x = int((spot["x"] * (DISPLAY_SIZE / 1024)) * self.zoom_level) + x_pos
            y = int((spot["y"] * (DISPLAY_SIZE / 1024)) * self.zoom_level) + y_pos
            self.canvas.create_oval(x-10, y-10, x+10, y+10, outline="yellow", width=3)
            self.lbl_status.config(text=f"진행: {len(self.current_answers)+1}/{self.total_dots} | [좌:0, 스페이스:1, q:취소]")
        else:
            self.save_frame_gt()

    def record_status(self, status):
        if self.current_file_path and len(self.current_answers) < self.total_dots:
            self.current_answers.append(status)
            self.display_image_and_dots()

    def undo_status(self):
        if self.current_answers:
            self.current_answers.pop()
            self.display_image_and_dots()

    def on_zoom(self, event):
        if event.num == 4 or event.delta > 0: zoom_change = 0.1
        else: zoom_change = -0.1
        self.zoom_level = min(max(self.zoom_level + zoom_change, 0.5), 4.0)
        self.display_image_and_dots()

    def start_pan(self, event): self.drag_start_x, self.drag_start_y = event.x, event.y
    def drag_pan(self, event):
        self.pan_x += event.x - self.drag_start_x
        self.pan_y += event.y - self.drag_start_y
        self.drag_start_x, self.drag_start_y = event.x, event.y
        self.display_image_and_dots()

    def save_frame_gt(self):
        self.load_master_gt()
        key = self.get_filename_key(self.current_file_path)
        self.master_gt[CAMERA_ID][key] = self.current_answers
        with open(OUTPUT_GT_JSON, "w", encoding="utf-8") as f:
            json.dump(self.master_gt, f, indent=2)
        messagebox.showinfo("저장", f"{key} 라벨링 완료!")

if __name__ == "__main__":
    root = tk.Tk()
    app = ParkingLabeler(root)
    root.mainloop()
