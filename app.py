import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from models.resnet_model import get_resnet50_model  
from class_mapping import class_mapping  

num_classes = len(class_mapping) 
model = get_resnet50_model(num_classes)

# Load the saved state_dict
model_path = './checkpoints/model_epoch_5.pth'
try:
    model.load_state_dict(torch.load(model_path))
    print("Model state_dict loaded successfully.")
except Exception as e:
    print(f"Error loading model state_dict: {e}")

# Set the model to evaluation mode
try:
    model.eval()
    print("Model set to evaluation mode.")
except Exception as e:
    print(f"Error setting model to evaluation mode: {e}")

# Define preprocessing steps
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

st.title('Model Inference GUI')

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image and make a prediction
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    # Get the predicted class name
    predicted_class_id = predicted.item()
    predicted_class_name = class_mapping.get(predicted_class_id, "Unknown class")

    st.write(f'Predicted class ID: {predicted_class_id}')
    st.write(f'Predicted class: {predicted_class_name}')
    
    # Debug: Print the output tensor
    st.write(f'Output tensor: {output}')
