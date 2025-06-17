import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import tkinter as tk
from tkinter import filedialog, Button, Label, Canvas, Scale, IntVar, Frame, HORIZONTAL, Radiobutton, Entry, StringVar, messagebox
from PIL import ImageTk
import os
import json
import base64
import io
import threading
import traceback
from openai import OpenAI

class AdvancedImageToCharacterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("高度な画像から象形文字ジェネレーター (ChatGPT連携版)")
        self.root.geometry("1000x800")
        
        self.input_image_path = None
        self.output_image = None
        self.processed_edges = None
        self.contours = None
        
        # APIキー関連
        self.api_key = StringVar()
        self.client = None
        
        # パラメータの初期値
        self.canny_threshold1 = IntVar(value=50)
        self.canny_threshold2 = IntVar(value=150)
        self.contour_simplification = IntVar(value=10)  # 0.01 * 1000 = 10
        self.line_thickness = IntVar(value=5)
        self.style_option = IntVar(value=0)  # 0: 輪郭のみ, 1: 塗りつぶし, 2: テクスチャ付き
        
        # APIキーの読み込み
        self.load_api_key()
        
        # UIの設定
        self.setup_ui()
    
    def load_api_key(self):
        try:
            if os.path.exists("api_key.json"):
                with open("api_key.json", "r") as f:
                    data = json.load(f)
                    self.api_key.set(data.get("api_key", ""))
                    if self.api_key.get():
                        self.client = OpenAI(api_key=self.api_key.get())
                        print("OpenAIクライアントを初期化しました")
        except Exception as e:
            print(f"APIキーの読み込みエラー: {str(e)}")
    
    def save_api_key(self):
        try:
            api_key = self.api_key.get()
            if api_key:
                with open("api_key.json", "w") as f:
                    json.dump({"api_key": api_key}, f)
                self.client = OpenAI(api_key=api_key)
                messagebox.showinfo("成功", "APIキーが保存されました")
                print("OpenAIクライアントを初期化しました")
            else:
                messagebox.showwarning("警告", "APIキーが入力されていません")
        except Exception as e:
            messagebox.showerror("エラー", f"APIキーの保存中にエラーが発生しました: {str(e)}")
    
    def setup_ui(self):
        # APIキー設定フレーム
        api_frame = tk.Frame(self.root)
        api_frame.pack(pady=5, fill=tk.X, padx=20)
        
        Label(api_frame, text="OpenAI APIキー:").pack(side=tk.LEFT, padx=5)
        api_entry = Entry(api_frame, textvariable=self.api_key, width=40, show="*")
        api_entry.pack(side=tk.LEFT, padx=5)
        
        Button(api_frame, text="保存", command=self.save_api_key).pack(side=tk.LEFT, padx=5)
        
        # 上部フレーム（ボタン用）
        top_frame = Frame(self.root)
        top_frame.pack(pady=10)
        
        # 画像選択ボタン
        self.select_btn = Button(top_frame, text="画像を選択", command=self.select_image)
        self.select_btn.pack(side=tk.LEFT, padx=10)
        
        # 変換ボタン
        self.convert_btn = Button(top_frame, text="象形文字に変換", command=self.convert_to_character)
        self.convert_btn.pack(side=tk.LEFT, padx=10)
        self.convert_btn.config(state=tk.DISABLED)
        
        # ChatGPTで象形文字生成ボタン
        self.chatgpt_btn = Button(top_frame, text="ChatGPTで象形文字生成", command=self.generate_with_chatgpt)
        self.chatgpt_btn.pack(side=tk.LEFT, padx=10)
        self.chatgpt_btn.config(state=tk.DISABLED)
        
        # 保存ボタン
        self.save_btn = Button(top_frame, text="画像を保存", command=self.save_image)
        self.save_btn.pack(side=tk.LEFT, padx=10)
        self.save_btn.config(state=tk.DISABLED)
        
        # 中央フレーム（画像表示用）
        self.center_frame = Frame(self.root)
        self.center_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 元画像表示用ラベル
        self.input_frame = Frame(self.center_frame)
        self.input_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.input_label = Label(self.input_frame, text="元の画像")
        self.input_label.pack()
        
        self.input_canvas = Canvas(self.input_frame, bg="lightgray")
        self.input_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 変換後画像表示用ラベル
        self.output_frame = Frame(self.center_frame)
        self.output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.output_label = Label(self.output_frame, text="象形文字")
        self.output_label.pack()
        
        self.output_canvas = Canvas(self.output_frame, bg="lightgray")
        self.output_canvas.pack(fill=tk.BOTH, expand=True)
        
        # パラメータ調整用フレーム
        param_frame = Frame(self.root)
        param_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Cannyエッジ検出のしきい値1
        Label(param_frame, text="エッジ検出 しきい値1:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        Scale(param_frame, from_=0, to=255, orient=HORIZONTAL, variable=self.canny_threshold1, 
              length=200).grid(row=0, column=1, padx=5, pady=2)
        
        # Cannyエッジ検出のしきい値2
        Label(param_frame, text="エッジ検出 しきい値2:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        Scale(param_frame, from_=0, to=255, orient=HORIZONTAL, variable=self.canny_threshold2, 
              length=200).grid(row=1, column=1, padx=5, pady=2)
        
        # 輪郭の単純化レベル
        Label(param_frame, text="輪郭の単純化レベル:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        Scale(param_frame, from_=1, to=100, orient=HORIZONTAL, variable=self.contour_simplification, 
              length=200).grid(row=0, column=3, padx=5, pady=2)
        
        # 線の太さ
        Label(param_frame, text="線の太さ:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        Scale(param_frame, from_=1, to=20, orient=HORIZONTAL, variable=self.line_thickness, 
              length=200).grid(row=1, column=3, padx=5, pady=2)
        
        # スタイルオプション
        style_frame = Frame(param_frame)
        style_frame.grid(row=2, column=0, columnspan=4, sticky=tk.W, padx=5, pady=5)
        
        Label(style_frame, text="スタイル:").pack(side=tk.LEFT, padx=5)
        Radiobutton(style_frame, text="輪郭のみ", variable=self.style_option, value=0).pack(side=tk.LEFT, padx=10)
        Radiobutton(style_frame, text="塗りつぶし", variable=self.style_option, value=1).pack(side=tk.LEFT, padx=10)
        Radiobutton(style_frame, text="テクスチャ付き", variable=self.style_option, value=2).pack(side=tk.LEFT, padx=10)
        
        # ChatGPTからの説明テキスト表示用フレーム
        self.description_frame = tk.Frame(self.root)
        self.description_frame.pack(fill=tk.X, padx=20, pady=5)
        
        Label(self.description_frame, text="ChatGPTからの応答:").pack(anchor=tk.W)
        
        self.description_text = tk.Text(self.description_frame, height=4, wrap=tk.WORD)
        self.description_text.pack(fill=tk.X, pady=5)
        self.description_text.config(state=tk.DISABLED)
        
        # ステータスバー
        self.status_label = Label(self.root, text="画像を選択してください", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="画像を選択",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.input_image_path = file_path
            self.display_input_image()
            self.convert_btn.config(state=tk.NORMAL)
            self.chatgpt_btn.config(state=tk.NORMAL if self.api_key.get() else tk.DISABLED)
            self.status_label.config(text=f"選択された画像: {os.path.basename(file_path)}")
    
    def display_input_image(self):
        # 入力画像を表示
        img = Image.open(self.input_image_path)
        img = self.resize_image_to_fit(img, self.input_canvas)
        
        self.input_photo = ImageTk.PhotoImage(img)
        self.input_canvas.config(width=img.width, height=img.height)
        self.input_canvas.create_image(0, 0, anchor=tk.NW, image=self.input_photo)
    
    def resize_image_to_fit(self, img, canvas, max_width=400, max_height=400):
        # キャンバスに合わせて画像をリサイズ
        width, height = img.size
        
        if width > max_width or height > max_height:
            ratio = min(max_width / width, max_height / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            return img.resize((new_width, new_height), Image.LANCZOS)
        
        return img
    
    def convert_to_character(self):
        if not self.input_image_path:
            return
        
        self.status_label.config(text="変換中...")
        self.root.update()
        
        # 画像から特徴を抽出し、象形文字を生成
        self.output_image = self.generate_character_from_image(self.input_image_path)
        
        # 結果を表示
        self.display_output_image()
        self.save_btn.config(state=tk.NORMAL)
        self.status_label.config(text="変換完了")
    
    def generate_with_chatgpt(self):
        if not self.input_image_path:
            return
        
        if not self.api_key.get() or not self.client:
            messagebox.showwarning("警告", "OpenAI APIキーが設定されていません。APIキーを入力して保存してください。")
            return
        
        try:
            self.status_label.config(text="ChatGPTに画像を送信中...")
            self.root.update()
            
            # 非同期で処理を実行
            threading.Thread(target=self._process_with_chatgpt, daemon=True).start()
            
        except Exception as e:
            self.status_label.config(text="エラーが発生しました")
            messagebox.showerror("エラー", f"ChatGPTとの通信中にエラーが発生しました: {str(e)}")
            print(f"エラー詳細: {traceback.format_exc()}")
    
    def _process_with_chatgpt(self):
        try:
            # 画像をBase64エンコード
            with open(self.input_image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
            
            # ChatGPTに画像を送信して象形文字を生成
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは画像から象形文字を生成する専門家です。与えられた画像の特徴を抽出し、その特徴を表現する象形文字を生成してください。象形文字はASCII文字またはUnicode文字で表現し、できるだけシンプルで特徴を捉えたものにしてください。また、生成した象形文字の説明も100文字程度で提供してください。"
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "この画像から象形文字を生成してください。象形文字はASCII文字またはUnicode文字で表現し、画像の特徴を捉えたシンプルなものにしてください。また、生成した象形文字の説明も提供してください。"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            # 応答を取得
            chatgpt_response = response.choices[0].message.content
            
            # UIスレッドで結果を表示
            self.root.after(0, lambda: self._update_ui_after_chatgpt(chatgpt_response))
            
        except Exception as error:
            # エラーメッセージを変数に保存してからラムダ関数に渡す
            error_message = str(error)
            self.root.after(0, lambda: self._show_error(error_message))
    
    def _update_ui_after_chatgpt(self, chatgpt_response):
        # 説明テキストを表示
        self.description_text.config(state=tk.NORMAL)
        self.description_text.delete(1.0, tk.END)
        self.description_text.insert(tk.END, chatgpt_response)
        self.description_text.config(state=tk.DISABLED)
        
        # 象形文字を抽出して表示
        try:
            # 象形文字を抽出（単純な例として、最初の行または特定のパターンを抽出）
            import re
            symbol_match = re.search(r'[^\w\s]|[\u2E80-\u9FFF]', chatgpt_response)
            
            # 象形文字を画像として表示
            canvas_size = (500, 500)
            character_img = Image.new('RGB', canvas_size, color='white')
            draw = ImageDraw.Draw(character_img)
            
            if symbol_match:
                symbol = symbol_match.group(0)
                
                # 中央に大きく描画
                font_size = 200
                try:
                    from PIL import ImageFont
                    # フォントを試す順番
                    font_names = ["Arial Unicode.ttf", "Arial.ttf", "DejaVuSans.ttf", "NotoSansCJK-Regular.ttc"]
                    font = None
                    
                    for font_name in font_names:
                        try:
                            font = ImageFont.truetype(font_name, font_size)
                            break
                        except:
                            continue
                except:
                    font = None
                
                # 中央に配置（テキストサイズの計算方法を更新）
                position = (canvas_size[0] // 2, canvas_size[1] // 2)
                
                # 描画
                if font:
                    # アンカーポイントを中央に設定
                    draw.text(position, symbol, fill='black', font=font, anchor="mm")
                else:
                    # フォントが利用できない場合は単純に中央付近に描画
                    draw.text((position[0] - 50, position[1] - 50), symbol, fill='black')
            else:
                # 象形文字が見つからない場合は、テキストから画像を生成
                lines = chatgpt_response.split('\n')
                y_position = 50
                
                for line in lines:
                    draw.text((50, y_position), line, fill='black')
                    y_position += 30
            
            self.output_image = character_img
            self.display_output_image()
            self.save_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            print(f"象形文字の表示中にエラーが発生しました: {str(e)}")
            print(traceback.format_exc())
            
            # エラーが発生しても、テキスト表示だけでも行う
            canvas_size = (500, 500)
            character_img = Image.new('RGB', canvas_size, color='white')
            draw = ImageDraw.Draw(character_img)
            
            lines = chatgpt_response.split('\n')
            y_position = 50
            
            for line in lines:
                draw.text((50, y_position), line, fill='black')
                y_position += 30
            
            self.output_image = character_img
            self.display_output_image()
            self.save_btn.config(state=tk.NORMAL)
        
        self.status_label.config(text="ChatGPTからの応答を受信しました")
    
    def _show_error(self, error_message):
        self.status_label.config(text="エラーが発生しました")
        messagebox.showerror("エラー", f"ChatGPTとの通信中にエラーが発生しました: {error_message}")
    
    def generate_character_from_image(self, image_path):
        # 画像を読み込み
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # エッジ検出（パラメータ調整可能）
        edges = cv2.Canny(gray, self.canny_threshold1.get(), self.canny_threshold2.get())
        self.processed_edges = edges.copy()
        
        # 輪郭検出
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = contours
        
        # 最も大きい輪郭を取得
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            
            # 輪郭を単純化（パラメータ調整可能）
            epsilon = (self.contour_simplification.get() / 1000) * cv2.arcLength(main_contour, True)
            approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)
            
            # 象形文字のような画像を生成
            canvas_size = (500, 500)
            character_img = Image.new('RGB', canvas_size, color='white')
            draw = ImageDraw.Draw(character_img)
            
            # 輪郭の座標を正規化して描画
            h, w = edges.shape
            scale_x = canvas_size[0] / w
            scale_y = canvas_size[1] / h
            
            # 中心に配置するためのオフセットを計算
            x_min = min(point[0][0] for point in approx_contour)
            y_min = min(point[0][1] for point in approx_contour)
            x_max = max(point[0][0] for point in approx_contour)
            y_max = max(point[0][1] for point in approx_contour)
            
            contour_width = x_max - x_min
            contour_height = y_max - y_min
            
            offset_x = (w - contour_width) // 2 - x_min
            offset_y = (h - contour_height) // 2 - y_min
            
            # 輪郭を描画
            points = []
            for point in approx_contour:
                x = int((point[0][0] + offset_x) * scale_x)
                y = int((point[0][1] + offset_y) * scale_y)
                points.append((x, y))
            
            # スタイルに応じて描画
            style = self.style_option.get()
            
            if style == 0:  # 輪郭のみ
                for i in range(len(points)):
                    start = points[i]
                    end = points[(i + 1) % len(points)]
                    draw.line([start, end], fill='black', width=self.line_thickness.get())
            
            elif style == 1:  # 塗りつぶし
                draw.polygon(points, outline='black', fill='black')
            
            elif style == 2:  # テクスチャ付き
                # まず輪郭を描画
                for i in range(len(points)):
                    start = points[i]
                    end = points[(i + 1) % len(points)]
                    draw.line([start, end], fill='black', width=self.line_thickness.get())
                
                # テクスチャ効果（ノイズや筆のストロークを模倣）
                texture_img = character_img.copy()
                draw_texture = ImageDraw.Draw(texture_img)
                
                # ポリゴン内部に短い線をランダムに描画してテクスチャを作成
                import random
                for _ in range(50):
                    x1 = random.randint(min(p[0] for p in points), max(p[0] for p in points))
                    y1 = random.randint(min(p[1] for p in points), max(p[1] for p in points))
                    x2 = x1 + random.randint(-30, 30)
                    y2 = y1 + random.randint(-30, 30)
                    draw_texture.line([(x1, y1), (x2, y2)], fill='black', width=1)
                
                # 元の画像とテクスチャをブレンド
                character_img = Image.blend(character_img, texture_img, 0.3)
                
                # 少しぼかして古い象形文字のような効果を追加
                character_img = character_img.filter(ImageFilter.GaussianBlur(0.5))
            
            return character_img
        else:
            # 輪郭が見つからない場合は元の画像をグレースケールで返す
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return pil_img
    
    def display_output_image(self):
        if self.output_image:
            # 出力画像をリサイズして表示
            img = self.resize_image_to_fit(self.output_image, self.output_canvas)
            
            self.output_photo = ImageTk.PhotoImage(img)
            self.output_canvas.config(width=img.width, height=img.height)
            self.output_canvas.create_image(0, 0, anchor=tk.NW, image=self.output_photo)
    
    def save_image(self):
        if not self.output_image:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="象形文字を保存",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            self.output_image.save(file_path)
            self.status_label.config(text=f"画像を保存しました: {os.path.basename(file_path)}")

def main():
    root = tk.Tk()
    app = AdvancedImageToCharacterApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()