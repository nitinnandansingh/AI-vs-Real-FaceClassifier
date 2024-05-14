import gradio as gr
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models

# Load your trained model state dictionary
model_path = 'resnet50_v3_model.pth'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Instantiate the model (assuming it's a ResNet-50)
model = models.resnet50(pretrained=False)
num_classes = 2  # Assuming 2 classes: Real Image and AI-generated Image
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Modify fully connected layer

# Load model state dictionary directly from checkpoint
model.load_state_dict(checkpoint)

# Set model to evaluation mode
model.eval()

# Define transformation for input images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def classify_image(image):
    # Preprocess the input image
    input_tensor = transform(Image.fromarray(image))
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    # Return prediction label
    return "Real Image" if predicted.item() == 0 else "AI-generated Image"

sample_image_1 = 'test_sample/ajay_fake.jpeg'
sample_image_2 = 'test_sample/salman_real.jpeg'
sample_image_3 = 'test_sample/srk_real.jpeg'

sample_image_4 = 'test_sample/ai_test.jpg'
sample_image_5 = 'test_sample/test1.jpg'
sample_image_6 = 'test_sample/test2.jpg'
sample_image_7 = 'test_sample/test3.jpg'
sample_image_8 = 'test_sample/test4.jpg'
sample_image_9 = 'test_sample/test5.jpg'
sample_image_10 = 'test_sample/test6.jpg'
sample_image_11 = 'test_sample/test7.jpg'

sample_images_list = [sample_image_1, 
                      sample_image_2, 
                      sample_image_3,
                      sample_image_4, 
                      sample_image_5, 
                      sample_image_6,
                      sample_image_7, 
                      sample_image_8, 
                      sample_image_9,
                      sample_image_10, 
                      sample_image_11]

gr.Interface(fn=classify_image,
             inputs=gr.Image(),
             outputs=gr.Label(num_top_classes=2),
             examples=(sample_images_list)).launch()