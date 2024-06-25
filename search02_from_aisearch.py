import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import pandas as pd
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_image(image_path):
    transform = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, image

def get_image_features(image_tensor):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    with torch.no_grad():
        features = model(image_tensor)
    return features.numpy().flatten()

def search_similar_images(search_service_name, index_name, api_key, image_features):
    endpoint = f"https://{search_service_name}.search.windows.net/"
    credential = AzureKeyCredential(api_key)
    client = SearchClient(endpoint, index_name, credential)

    try:
        results = client.search(search_text="", filter=None)
        results_data = []
        for result in results:
            if 'ImageId' in result and 'Vector' in result:
                vector = np.array(result['Vector'])
                similarity = cosine_similarity([image_features], [vector])[0][0]
                results_data.append({'Image ID': result['ImageId'], 'Similarity': similarity})
        if results_data:
            results_df = pd.DataFrame(results_data)
            print("Search Results:")
            print(results_df)
            display_results_graphically(results_df)
        else:
            print("No results found.")
    except Exception as e:
        print(f"An error occurred while searching: {e}")

def display_results_graphically(results_df):
    if not results_df.empty:
        results_df.set_index('Image ID')['Similarity'].plot(kind='bar', color='skyblue')
        plt.title('Similarity Scores')
        plt.xlabel('Image ID')
        plt.ylabel('Similarity')
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("No results to display graphically.")

if __name__ == "__main__":
    image_path = input("Please enter the path to your image: ").strip()  # Remove leading and trailing spaces
    search_service_name = 'ajayaisearch01'
    index_name = 'aisearchindex02'
    api_key = 'xxx'

    image_tensor, _ = load_image(image_path)
    image_features = get_image_features(image_tensor)

    search_similar_images(search_service_name, index_name, api_key, image_features)
