import clip
import torch
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load('ViT-B/32', device=device)

def find_top3_images(text_query, image_paths):
    # Tokenizing 
    text_tokens = clip.tokenize([text_query]).to(device)

    # Images processing
    images = [preprocess(Image.open(path)).unsqueeze(0) for path in image_paths]
    image_input = torch.cat(images).to(device)

    # Embedding without grads
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)

    # Embeddings ormalizing
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Similarities calculation
    similarities = (image_features @ text_features.T).squeeze()

    # Finding top 3 scores
    top3_scores, top3_indices = torch.topk(similarities, 3)

    # Preparing top 3 paths and scores
    top3_image_paths = [image_paths[i] for i in top3_indices]
    top3_scores_list = top3_scores.tolist()

    return top3_image_paths, top3_scores_list

image_folder = "pics"
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]

query = "a photo of a flower"

best_images, scores = find_top3_images(query, image_paths)
print("Top 3 images and similarity:")
for path, score in zip(best_images, scores):
    print(f"{path}: {score:.4f}")

