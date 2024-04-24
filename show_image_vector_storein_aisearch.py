import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import re
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient, IndexDocumentsBatch

# Define Azure Cognitive Search parameters
search_service_name = 'ajayaisearch01'
index_name = 'aisearchindex02'
api_key = 'R0wUGK9f8L1MNXz8v7Z4Rs2JHNKuClOHwmOATgIjzuAzSeAxwqlS'
endpoint = f"https://{search_service_name}.search.windows.net/"
credential = AzureKeyCredential(api_key)

search_client = SearchClient(endpoint, index_name, credential)

def upload_vector_to_search(image_id, vector):
    batch = IndexDocumentsBatch()
    document = {
        "ImageId": sanitize_key(image_id),
        "Vector": vector.tolist()
    }
    batch.add_upload_actions([document])
    try:
        result = search_client.index_documents(batch)
        print("Upload to Azure Search successful:", result)
    except Exception as e:
        print("Failed to upload document to Azure Search:", e)

def get_image_embedding(image_path, model=resnet50(weights=ResNet50_Weights.DEFAULT)):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model.eval()
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img)
        batch_t = torch.unsqueeze(img_t, 0)
        with torch.no_grad():
            embedding = model(batch_t)
        return embedding.numpy().flatten()
    except IOError as e:
        print(f"Error opening image {image_path}: {e}")
        return None

def sanitize_key(key):
    """Sanitizes the key to conform to Azure Search requirements."""
    return re.sub(r'[^a-zA-Z0-9_\-=]', '_', key)

image_paths = [
    "E:\\code\\oracle\\logo\\amazon.jfif",
    "E:\\code\\oracle\\logo\\bajaj.jfif",
    "E:\\code\\oracle\\logo\\google.jfif",
    "E:\\code\\oracle\\logo\\hero.jfif",
    "E:\\code\\oracle\\logo\\indigo.jfif",
    "E:\\code\\oracle\\logo\\microsoft.jfif",
    "E:\\code\\oracle\\logo\\tcs.jfif",
    "E:\\code\\oracle\\logo\\wipro.jfif"
]

if __name__ == "__main__":
    for path in image_paths:
        vector = get_image_embedding(path)
        if vector is not None:
            image_id = path.split('\\')[-1]
            upload_vector_to_search(image_id, vector)
        else:
            print(f"Failed to process vector for {path}.")
