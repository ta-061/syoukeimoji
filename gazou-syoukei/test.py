#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import json
import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
import openai
import requests
import time
import datetime

class HieroglyphApp:
    def __init__(self):
        # api_key.json から APIキー を取得
        key_file = os.path.join(os.getcwd(), "api_key.json")
        if not os.path.isfile(key_file):
            raise RuntimeError(f"api_key.json が見つかりません: {key_file}")
        with open(key_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        api_key = data.get('api_key')
        if not api_key:
            raise RuntimeError("api_key.json に 'api_key' が定義されていません")
        openai.api_key = api_key

        self.root = tk.Tk()
        self.root.title("象形文字変換アプリ")
        self.root.geometry("600x700")
        
        # ── 上部：画像表示用 Canvas ──
        frame_imgs = tk.Frame(self.root)
        frame_imgs.pack(pady=10)
        self.canvas_input = tk.Canvas(frame_imgs, width=256, height=256, bg="#eee", highlightthickness=1, highlightbackground="black")
        self.canvas_input.pack(side=tk.LEFT, padx=10)
        self.canvas_output = tk.Canvas(frame_imgs, width=256, height=256, bg="#eee", highlightthickness=1, highlightbackground="black")
        self.canvas_output.pack(side=tk.LEFT, padx=10)
        
        # ── 中段：操作ボタン ──
        frame_btn = tk.Frame(self.root)
        frame_btn.pack(pady=5)
        btn_load = tk.Button(frame_btn, text="画像を読み込み", command=self.load_image)
        btn_load.pack(side=tk.LEFT, padx=5)
        self.btn_convert = tk.Button(frame_btn, text="変換", state=tk.DISABLED, command=self.convert_image)
        self.btn_convert.pack(side=tk.LEFT, padx=5)
        
        # ── 下部：ログ出力用テキスト ──
        self.log = scrolledtext.ScrolledText(self.root, width=70, height=20)
        self.log.pack(pady=10)
        
        # 状態保持
        self.input_path = None
        self.input_image = None

    def log_print(self, msg: str):
        """テキストボックスに追記＆自動スクロール"""
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.root.update()

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("JPEG ファイル", ("*.jpg", "*.jpeg")),
                ("PNG ファイル", "*.png"),
                ("BMP ファイル", "*.bmp"),
                ("WebP ファイル", "*.webp"),
                ("すべての画像", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")),
                ("すべてのファイル", "*.*"),
            ])
        if not path:
            return
        img = Image.open(path)
        self.input_image = img.copy()
        self.input_path = path
        self.display_on_canvas(img, self.canvas_input)
        self.btn_convert.config(state=tk.NORMAL)
        self.log_print(f"[読み込み] {os.path.basename(path)}")

    def display_on_canvas(self, pil_img, canvas):
        w, h = pil_img.size
        pil_img = pil_img.copy()
        pil_img.thumbnail((256, 256), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.image = tk_img
        canvas.delete("all")
        canvas.create_image(128, 128, image=tk_img)

    def convert_image(self):
        if not self.input_path:
            return
        self.log_print("[開始] 画像を象形文字に変換中…")
        start_time = time.time()
        self.log_print(f"[{datetime.datetime.now()}] convert_image メソッド開始")
        
        mask = Image.new("RGBA", self.input_image.size, (255, 255, 255, 255))
        mask_path = os.path.join(os.getcwd(), "mask.png")
        mask.save(mask_path)
        
        api_start = time.time()
        self.log_print(f"[{datetime.datetime.now()}] API呼び出し送信")
        try:
            with open(self.input_path, "rb") as img_f, open(mask_path, "rb") as mask_f:
                resp = openai.Image.create_edit(
                    image=img_f,
                    mask=mask_f,
                    prompt="象形文字の画像を生成してください",
                    n=1,
                    size="512x512"
                )
                api_end = time.time()
                self.log_print(f"[{datetime.datetime.now()}] API応答受信 ({api_end - api_start:.2f}秒)")
        except Exception as e:
            self.log_print(f"[エラー] API呼び出しに失敗しました: {e}")
            return
        
        url = resp["data"][0]["url"]
        self.log_print(f"[完了] 生成画像のURL取得: {url}")
        
        dl_start = time.time()
        self.log_print(f"[{datetime.datetime.now()}] 画像ダウンロード開始")
        try:
            r = requests.get(url)
            r.raise_for_status()
            dl_end = time.time()
            self.log_print(f"[{datetime.datetime.now()}] 画像ダウンロード完了 ({dl_end - dl_start:.2f}秒)")
        except Exception as e:
            self.log_print(f"[エラー] 画像ダウンロード失敗: {e}")
            return
        
        gen_img = Image.open(io.BytesIO(r.content))
        out_path = os.path.join(os.getcwd(), "generated.png")
        gen_img.save(out_path)
        self.log_print(f"[保存] {out_path}")
        
        self.display_on_canvas(gen_img, self.canvas_output)
        self.log_print("[完了] ウィンドウに表示しました")
        total_end = time.time()
        self.log_print(f"[{datetime.datetime.now()}] convert_image メソッド完了 (合計 {(total_end - start_time):.2f}秒)")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = HieroglyphApp()
    app.run()
