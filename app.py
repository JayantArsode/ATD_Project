from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from flask import Flask, render_template, request
import base64
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Define labels
labels = ['Bowel Healthy', 'Bowel Injury', 'Extravasation Healthy', 'Extravasation Injury', 'Kidney Healthy',
          'Kidney Low', 'Kidney High', 'Liver Healthy', 'Liver Low', 'Liver High', 'Spleen Healthy', 'Spleen Low', 'Spleen High']
# Define the ATD_ResNet model class


class ATD_EfficientNet(nn.Module):
    def __init__(self, input_shape=3, model_name='efficientnet-b1', num_blocks_to_unfreeze=0):
        super().__init__()
        self.efficientnet_base = EfficientNet.from_pretrained(model_name, in_channels=3, image_size=[
                                                              256, 256])  # Assuming pretrained with ImageNet weights

        # Freeze all parameters of EfficientNet if pretrained is True
        for param in self.efficientnet_base.parameters():
            param.requires_grad = False

        # Unfreeze specific blocks of EfficientNet for fine-tuning
        if num_blocks_to_unfreeze > 0:
            for param in self.efficientnet_base._blocks[-num_blocks_to_unfreeze:].parameters():
                param.requires_grad = True

        # Get the number of features from the last layer of EfficientNet
        num_features = self.efficientnet_base._fc.in_features
        self.flatten = nn.Flatten()  # Flatten the output of EfficientNet
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(
            1)  # Global average pooling

        self.bowel_head = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

        self.extra_head = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )

        self.liver_head = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.SiLU(),
            nn.Linear(64, 3)
        )

        self.kidney_head = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.SiLU(),
            nn.Linear(64, 3)
        )

        self.spleen_head = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.SiLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.flatten(self.adaptive_pooling(
            self.efficientnet_base.extract_features(x)))
        bowel_out = self.bowel_head(x)
        extra_out = self.extra_head(x)
        kidney_out = self.kidney_head(x)
        liver_out = self.liver_head(x)
        spleen_out = self.spleen_head(x)
        return bowel_out, extra_out, kidney_out, liver_out, spleen_out


# Load the pre-trained model
model = ATD_EfficientNet(
    input_shape=3, model_name='efficientnet-b1', num_blocks_to_unfreeze=4)
model.load_state_dict(torch.load(
    'ATD_EfficientNetB1_CutMixUp_Model.pth', map_location=torch.device('cpu')))
model.eval()

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Function to preprocess the image


def preprocess_image(image_path):
    images = []  # Initialize empty list
    image = Image.open(image_path).convert('RGB')
    # Assuming test_transform is defined elsewhere
    image = transform(image) / 255.0
    images.append(image)  # Append image to list

    return torch.stack(images)

# Function to classify the image


def predictions(images, model_instance):
    with torch.inference_mode():
        y_bowel, y_extra, y_kidney, y_liver, y_spleen = model_instance(images)
    y_bowel = torch.sigmoid(y_bowel)
    y_extra = torch.sigmoid(y_extra)
    y_kidney = torch.softmax(y_kidney, dim=1)
    y_liver = torch.softmax(y_liver, dim=1)
    y_spleen = torch.softmax(y_spleen, dim=1)

    bowel_injury = y_bowel.numpy().flatten().item()
    bowel_healthy = 1-bowel_injury

    extra_injury = y_extra.numpy().flatten().item()
    extra_healthy = 1-extra_injury

    y_kidney = y_kidney.numpy()
    kidney_healthy = y_kidney[:, 0].item()
    kidney_low = y_kidney[:, 1].item()
    kidney_high = y_kidney[:, 2].item()

    y_liver = y_liver.numpy()
    liver_healthy = y_liver[:, 0].item()
    liver_low = y_liver[:, 1].item()
    liver_high = y_liver[:, 2].item()

    y_spleen = y_spleen.numpy()
    spleen_healthy = y_spleen[:, 0].item()
    spleen_low = y_spleen[:, 1].item()
    spleen_high = y_spleen[:, 2].item()

    return [bowel_healthy, bowel_injury, extra_healthy, extra_injury, kidney_healthy, kidney_low, kidney_high, liver_healthy, liver_low, liver_high, spleen_healthy, spleen_low, spleen_high]


@app.route('/', methods=['GET'])
def upload():
    return render_template('upload.html')


@app.route('/generate', methods=['POST'])
def generate():
    image_file = request.files.get('image', None)
    if image_file:
        image_tensor = preprocess_image(image_file)
        probabilities = predictions(image_tensor, model)
        image_file.seek(0)  # Reset file pointer to start
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
        return render_template('report.html', image_data=image_data, probabilities=probabilities, labels=labels)
    return redirect(url_for('upload'))


if __name__ == '__main__':
    app.run(debug=True)
