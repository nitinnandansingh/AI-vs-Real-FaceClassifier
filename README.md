# AI-vs-Real Face Classifier

This project aims to classify images as either real human faces or AI-generated faces using a pre-trained ResNet-50 model fine-tuned for this specific task.

## Overview

With the increasing sophistication of AI-generated images, it's crucial to develop models that can distinguish between real and AI-generated faces. This project leverages a ResNet-50 model fine-tuned to perform this classification task.

## Installation

To get started with the project, follow these steps:

1. **Clone the repository**:
    ```sh
    git clone https://github.com/nitinnandansingh/AI-vs-Real-FaceClassifier.git
    cd AI-vs-Real-FaceClassifier
    ```

2. **Set up a virtual environment** (optional but recommended):
    ```sh
    python3 -m venv aiimageclassifier
    source aiimageclassifier/bin/activate
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run the Gradio app and classify images, execute the following command:
```sh
python app.py
```

To run it on HuggingFace Spaces and classify images, execute the following command:
```sh
https://huggingface.co/spaces/nitinnsingh/AI-vs-Real-FaceDetector
```



