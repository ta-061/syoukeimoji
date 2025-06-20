# 画像から象形文字ジェネレーター (ChatGPT版)

このアプリケーションは、入力された画像から特徴を抽出し、象形文字のような画像を生成します。さらに、ChatGPT APIを使用して、生成された象形文字の説明文を自動的に作成します。

## 機能

- 画像ファイルの選択
- 画像から特徴（輪郭）の抽出
- ChatGPT APIを使用した象形文字の説明文生成
- 象形文字のような画像の生成
- 生成された画像の保存

## 必要条件

- Python 3.6以上
- OpenCV
- NumPy
- Pillow (PIL)
- OpenAI Python ライブラリ
- OpenAI APIキー

## インストール方法

1. リポジトリをクローンまたはダウンロードします
2. 必要なライブラリをインストールします：

```
pip install -r requirements.txt
```

## 使用方法

1. アプリケーションを起動します：

```
python fixed_main.py
```

2. OpenAI APIキーを入力して「保存」ボタンをクリックします
3. 「画像を選択」ボタンをクリックして、変換したい画像を選択します
4. 「象形文字に変換」ボタンをクリックして、画像を変換します
5. 生成された象形文字と説明文が表示されます
6. 「画像を保存」ボタンをクリックして、生成された象形文字画像を保存します

## 仕組み

1. 入力画像からエッジを検出します（Cannyエッジ検出）
2. 検出されたエッジから輪郭を抽出します
3. 最も大きな輪郭を選択し、単純化します
4. 輪郭の特徴（点の数、面積、周囲長など）を抽出します
5. 抽出した特徴とファイル名をChatGPT APIに送信して、象形文字の説明文を生成します
6. 単純化された輪郭を使用して、象形文字のような画像を生成します
7. 生成された象形文字と説明文を表示します

## OpenAI APIキーについて

このアプリケーションを使用するには、OpenAI APIキーが必要です。APIキーは以下の手順で取得できます：

1. [OpenAIのウェブサイト](https://platform.openai.com/)にアクセスしてアカウントを作成します
2. APIキーを生成します
3. アプリケーション起動後、APIキー入力欄にキーを入力して「保存」ボタンをクリックします

APIキーは暗号化されずにローカルファイル（api_key.json）に保存されますので、取り扱いにご注意ください。