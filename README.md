# 🔐 AI-Based Intrusion Detection and Prevention System (IDPS)

An **AI-powered Intrusion Detection and Prevention System** that converts network traffic data into images, detects malicious activities using a **Lightweight CNN**, and improves robustness using **Adversarial Attack Generation (FGSM) and Adversarial Training**.

This project is designed for **academic research, final-year projects, and SOC/security-oriented applications**.

---

## 📌 Project Highlights

* Uses **CIC IoT 2023 Dataset**
* Converts network flow features into **image representations**
* Lightweight **CNN-based IDS**
* **FGSM adversarial attack generation**
* **Adversarial training for attack prevention**
* High accuracy and robustness against adversarial perturbations
* Fully reproducible execution pipeline

---

## 🧠 Project Architecture (High Level)

```
Dataset (CSV)
   ↓
Preprocessing & Normalization
   ↓
Feature-to-Image Conversion
   ↓
CNN Training (Normal IDS)
   ↓
FGSM Adversarial Attack
   ↓
Adversarial Training (Defense)
   ↓
Evaluation & Results
```

---

## 📂 Project Directory Structure

```
IDPS/
│
├── preprocess.py              # Dataset loading, preprocessing, image generation
├── train_cnn.py               # Lightweight CNN model training
├── fgsm_attack.py             # FGSM adversarial attack generation
├── adversarial_training.py    # Adversarial training (attack prevention)
├── result.py                  # Model evaluation and results
│
├── IDPS_Output/
│   └── Binary_IDS/
│       ├── images/
│       │   ├── Normal/
│       │   └── Attack/
│       ├── scaler.pkl
│       ├── feature_order.txt
│       └── features.csv
│
├── cnn_ids_model.keras        # Trained CNN IDS model
├── cnn_ids_adv_trained.keras  # Adversarially trained IDS model
└── README.md
```

---

## 📊 Dataset Used

* **Dataset:** CIC IoT 2023
* **Source:**
  [https://www.unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
* **Features:** 39 network flow features
* **Classes Used:**

  * Normal
  * Attack

---

## ⚙️ Requirements

Install required libraries using:

```bash
pip install tensorflow numpy pandas scikit-learn pillow joblib
```

> ⚠️ Recommended Python version: **Python 3.9+**

---

## ▶️ Execution Steps (IMPORTANT)

Follow the steps **in exact order** 👇

---

### **Step 1: Preprocess Normal Traffic**

```bash
python preprocess.py
```

* Select **Normal** when prompted
* Generates Normal traffic images

Output:

```
🖼 Normal: 1000 images generated
```

---

### **Step 2: Preprocess Attack Traffic**

```bash
python preprocess.py
```

* Select **Attack** when prompted
* Generates Attack traffic images

Output:

```
🖼 Attack: 1000 images generated
```

---

### **Step 3: Train CNN IDS Model**

```bash
python train_cnn.py
```

* Trains lightweight CNN
* Performs 80–20 train-validation split
* Saves trained model

Output file:

```
cnn_ids_model.keras
```

---

### **Step 4: Evaluate IDS Performance (Before Attack)**

```bash
python result.py
```

* Displays:

  * Accuracy
  * Precision, Recall, F1-score
  * Confusion Matrix

---

### **Step 5: Generate Adversarial Attacks (FGSM)**

```bash
python fgsm_attack.py
```

* Implements FGSM attack using gradient-based perturbations
* Epsilon (ε) = 0.02

Output:

```
⚠️ FGSM attack logic ready
```

---

### **Step 6: Adversarial Training (Attack Prevention)**

```bash
python adversarial_training.py
```

* Combines original + adversarial samples
* Retrains CNN for robustness
* Saves protected IDS model

Output file:

```
cnn_ids_adv_trained.keras
```

---

### **Step 7: Evaluate IDS After Attack Prevention**

```bash
python result.py
```

* Confirms robustness of IDS
* Compares accuracy before and after defense

---

## 📈 Sample Results

* **Accuracy:** ~99%
* **False Positives:** Very Low
* **False Negatives:** Near Zero
* **Strong resistance to FGSM adversarial attacks**

Example Confusion Matrix:

```
[[ 983   17]
 [   0 1000]]
```

---

## 🛡️ Security Techniques Used

* Feature normalization (Min-Max Scaling)
* CNN-based pattern recognition
* FGSM (Fast Gradient Sign Method)
* Adversarial Training
* Early stopping & learning rate scheduling

---

## 🎓 Academic & Practical Use Cases

* Final-year engineering project
* Research on adversarial ML security
* SOC Analyst skill demonstration
* AI-based Network Security Systems
* IDS robustness analysis

---

## 🚀 Future Enhancements

* CNN + LSTM hybrid model (temporal traffic analysis)
* Multi-class attack classification
* PGD / BIM adversarial attacks
* Explainable AI (Grad-CAM)
* Real-time traffic simulation
* SIEM / SOC integration

---


## 📌 Author

**Yash Somvanshi**
Cybersecurity | AI | IDS | Adversarial Machine Learning

GitHub: [https://github.com/somvanshiyash46](https://github.com/somvanshiyash46)

---
