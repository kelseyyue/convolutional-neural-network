# convolutional-neural-network
Image Recognition

# **Flower Classification using Transfer Learning**

This project demonstrates a deep learning-based image classification pipeline to classify flower images into 102 categories using transfer learning with the ResNet-18 architecture.

## **Features**
- Pretrained ResNet-18 model fine-tuned for flower classification.
- Data augmentation techniques for improving generalization.
- GPU-accelerated training and evaluation.
- Visualization of predictions with color-coded accuracy indicators.

---

## **Project Structure**
```
.
├── flower_data/                 # Dataset directory
│   ├── train/                   # Training dataset
│   └── valid/                   # Validation dataset
├── cat_to_name.json             # Mapping of category indices to flower names
└── script.py                    # Main Python script for training and evaluation
```

---

## **Dependencies**
Ensure you have the following libraries installed:
- `torch`
- `torchvision`
- `numpy`
- `matplotlib`
- `Pillow`
- `imageio`

Install them using:
```bash
pip install torch torchvision numpy matplotlib pillow imageio
```

---

## **Usage**

### **1. Dataset Preparation**
- Place the flower dataset in the `flower_data` directory with the following structure:
  ```
  flower_data/
  ├── train/
  │   ├── class1/
  │   ├── class2/
  │   └── ...
  └── valid/
      ├── class1/
      ├── class2/
      └── ...
  ```
- Ensure you have a `cat_to_name.json` file that maps category indices to human-readable flower names.

### **2. Training**
Run the script to train the model:
```bash
python script.py
```
- The model will save the best weights to `best.pt` during training.

### **3. Loading a Trained Model**
To load and evaluate the best model:
```python
checkpoint = torch.load('best.pt')
model_ft.load_state_dict(checkpoint['state_dict'])
```

### **4. Predict and Visualize**
The script includes functionality to predict on the validation dataset and visualize the results. Example:
```python
dataiter = iter(dataloaders['valid'])
images, labels = dataiter.next()
output = model_ft(images.cuda() if torch.cuda.is_available() else images)
```
The visualization highlights correct predictions in green and incorrect predictions in red.

---

## **Model Pipeline**

1. **Data Preprocessing**:
   - Training:
     - Resize to `[96x96]`
     - Data augmentations: rotation, cropping, horizontal and vertical flips, color jitter.
   - Validation:
     - Resize to `[64x64]`.

2. **Model**:
   - Pretrained **ResNet-18** with the final fully connected layer replaced to classify 102 categories.
   - Option to freeze pre-trained layers during feature extraction.

3. **Training**:
   - Optimizer: Adam with learning rate `1e-2`.
   - Scheduler: StepLR (reduces learning rate by 10x every 10 epochs).
   - Loss: CrossEntropyLoss.
   - Tracks training/validation accuracy and loss for performance monitoring.

4. **Evaluation**:
   - Predicts the class for validation images and compares predictions to ground truth.
   - Visualizes predictions with the flower name and color-coded correctness.

---

## **Visualization**
- The project includes a visualization tool that displays the images, predicted flower names, and ground truth labels.
- Correct predictions are shown in **green**, incorrect predictions in **red**.

---

## **Results**
- **Best Validation Accuracy**: Achieved during training and saved in `best.pt`.
- Example visualization:
![Visualization Example](#) *(Replace with your image)*

---

## **Future Improvements**
- Experiment with deeper architectures like ResNet-50 or EfficientNet.
- Add support for real-time prediction on new images.
- Fine-tune hyperparameters for further improvement in accuracy.

---

## **License**
This project is for educational purposes. Please ensure proper attribution if used elsewhere.

--- 

将此文件保存为 `README.md`，可用于展示你的项目！如果需要调整或添加更多内容，请告诉我！
