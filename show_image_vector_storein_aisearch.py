# az search index list --service-name ajayaisearch01 --resource-group your-resource-group-name --output table

import torch
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from PIL import Image
import re
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient, IndexDocumentsBatch
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchFieldDataType

# Define Azure Cognitive Search parameters
search_service_name = 'xxxxx'
index_name = 'xxxxxxx'
api_key = 'xxxxxxx'
endpoint = f"https://{search_service_name}.search.windows.net/"
credential = AzureKeyCredential(api_key)

# Create or verify the index
def create_or_verify_index():
    client = SearchIndexClient(endpoint, credential)
    try:
        # Try to get the index if it exists
        client.get_index(index_name)
        print("Index already exists.")
    except Exception as e:
        # If not, create the index
        print("Index does not exist. Creating index...")
        index = SearchIndex(
            name=index_name,
            fields=[
                SimpleField(name="ImageId", type=SearchFieldDataType.String, key=True, filterable=True),
                SimpleField(name="Vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Double), searchable=False)
            ]
        )
        client.create_index(index)
        print("Index created.")

# Initialize search client
search_client = SearchClient(endpoint, index_name, credential)

# Existing functions...
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

# Main function
if __name__ == "__main__":
    create_or_verify_index()
    # List of image paths
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
    for path in image_paths:
        vector = get_image_embedding(path)
        if vector is not None:
            image_id = path.split('\\')[-1]
            upload_vector_to_search(image_id, vector)
        else:
            print(f"Failed to process vector for {path}.")
