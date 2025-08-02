# 🎨🔍 CLIP Multimodal Image Search

Welcome to **CLIP Multimodal Image Search** — a sleek Python tool that leverages OpenAI's **CLIP** model to match images with text queries! 🚀  
Harness the power of multimodal AI to find the best matching images based on your textual input. Perfect for exploring semantic similarity across vision and language! 🧠✨

---

## ⚡ Features

- 🔥 Uses state-of-the-art **CLIP (ViT-B/32)** model for embedding images and text  
- 🖼️ Batch processes multiple images effortlessly  
- 📏 Calculates cosine similarity to find top matching images  
- 🎯 Easily extendable to support top-k matches and more image formats  
- 💻 GPU accelerated if available for faster processing

---

## 🛠️ Requirements

- Python 3.7+  
- [PyTorch](https://pytorch.org/)  
- [OpenAI CLIP](https://github.com/openai/CLIP)  
- [Pillow](https://python-pillow.org/)  

Install all dependencies with:  
bash
pip install -r requirements.txt

## 🚀 Usage

1. Add your images (JPG or PNG) to the pics/ folder 📂
2. Modify the text_query variable in the script with your search phrase ✍️
3. Run the script and watch magic happen:
python find_top3_images.py

Get the best matching images based on your query — easy and fast! ⚡

## 📁 Project Structure

clip-multimodal-search/
├── pics/                  # Sample images folder  
├── find_top3_images.py    # Main CLIP search script  
├── requirements.txt       # Python dependencies  
└── README.md              # This awesome doc!  



