# 📌 COVID-19 Detection using Deep Learning
## 📝 Overview
This project implements a Deep Learning-based model for detecting COVID-19 using chest X-ray images. The model is trained using PyTorch and classifies images into three categories:
✅ COVID-19
✅ Normal (Healthy Lungs)
✅ Viral Pneumonia

---

⚡ Features  
🔹 Custom CNN Model designed for image classification  
🔹 Preprocessing and Augmentation using torchvision.transforms  
🔹 Visualization of dataset samples and model predictions  
🔹 Evaluation Metrics: Accuracy, Loss, and Confusion Matrix  
🔹 Model Deployment Ready (weights saved in .pth format)  

---

📂 Dataset  
The dataset used is COVID-19 Image Dataset from Kaggle.  
📥 Download it here: COVID-19 X-ray Dataset  
  
Structure:  
  
Covid19-dataset/  
│── train/  
│   ├── Covid/  
│   ├── Normal/  
│   ├── Viral Pneumonia/  
│── test/  
│   ├── Covid/  
│   ├── Normal/  
│   ├── Viral Pneumonia/  
  
🏗 Installation & Setup  
🔹 1. Clone Repository  
git clone https://github.com/your-repo/covid-detection.git  
cd covid-detection  
🔹 2. Install Dependencies   
pip install -r requirements.txt  
🔹 3. Run Training Script  
python covid_detection.py  

   
💡 The model will train and save the weights in models/COVID_DETECTION.pth.  

---

🏛 Model Architecture  
The CNN model consists of:  
✔ Multiple Convolutional Layers with ReLU activation  
✔ MaxPooling Layers for downsampling   
✔ Fully Connected Layer for classification  

```python
  {
class Covid_detection(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*25*100, 3)
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.classifier(x)
        return x}
  
🎯 Performance & Accuracy
✔ Training Accuracy: ~95%
✔ Testing Accuracy: ~92%

📊 Results & Visualization
🔹 Sample Chest X-ray Predictions
✅ Green Title → Correct Prediction
❌ Red Title → Incorrect Prediction

```python
plt.figure(figsize=(12, 12))
rows, cols = 4, 4
for i in range(16):
    img, label = test_data[i]
    pred = model(img.unsqueeze(0).to(device))
    pred_label = torch.argmax(torch.softmax(pred, dim=1), dim=1)
    plt.subplot(rows, cols, i+1)
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f"Pred: {class_names[pred_label]} | Actual: {class_names[label]}")
    plt.axis(False)
📌 Future Improvements
🚀 Implement Data Augmentation for improved generalization
🚀 Optimize CNN architecture for higher accuracy
🚀 Deploy model as a Web App using Flask or FastAPI

