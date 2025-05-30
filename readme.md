# სახის გამომეტყველების ამოცნობის გამოწვევა

ეს რეპოზიტორია მოიცავს ჩემს იმპლემენტაციას Kaggle-ის [Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge) კონკურსისთვის. პროექტის მიზანია ღრმა სწავლის მოდელების შექმნა და შეფასება, რომლებიც კლასიფიცირებენ სახის გამომეტყველებებს შვიდი სხვადასხვა ემოციის კატეგორიად.

## 📋 შინაარსი

* [პროექტის მიმოხილვა](#-პროექტის-მიმოხილვა)
* [მონაცემთა ნაკრები](#-მონაცემთა-ნაკრები)
* [დაყენება და ინსტალაცია](#-დაყენება-და-ინსტალაცია)
* [პროექტის სტრუქტურა](#-პროექტის-სტრუქტურა)
* [მეთოდოლოგია](#-მეთოდოლოგია)
* [ექსპერიმენტები და შედეგები](#-ექსპერიმენტები-და-შედეგები)
* [Weights & Biases ინტეგრაცია](#-weights--biases-ინტეგრაცია)
* [მიგნებები და ანალიზი](#-მიგნებები-და-ანალიზი)
* [გამოყენება](#-გამოყენება)
* [შეუერთდი პროექტს](#-შეუერთდი-პროექტს)
* [ლიცენზია](#-ლიცენზია)

## 🌟 პროექტის მიმოხილვა

ეს პროექტი იკვლევს სხვადასხვა ნეირონულ ქსელის არქიტექტურას სახის გამომეტყველების ამოსაცნობად, ყურადღება გამახვილებულია ჰიპერპარამეტრებისა და არქიტექტურული გადაწყვეტილებების გავლენაზე მოდელის მუშაობაზე. პროექტი აგებულია PyTorch-ის გამოყენებით, ხოლო ყველა ექსპერიმენტი ლოგირდება Weights & Biases პლატფორმაზე სრული ანალიზისა და ვიზუალიზაციისთვის.

## 📊 მონაცემთა ნაკრები

მონაცემთა ნაკრები შედგება 48x48 პიქსელის გრეისკეილ სახეების გამოსახულებებისგან, თითოეული ლეიბლირებულია ერთ-ერთი შემდეგი ემოციის კატეგორიით:

* 0: გაბრაზება (Angry)
* 1: ზიზღი (Disgust)
* 2: შიში (Fear)
* 3: სიხარული (Happy)
* 4: სევდა (Sad)
* 5: გაოცება (Surprise)
* 6: ნეიტრალური (Neutral)

## 🛠️ დაყენება და ინსტალაცია

1. დააკლონეთ რეპოზიტორია:

   ```bash
   git clone https://github.com/your-username/Facial-Expression-Recognition.git
   cd Facial-Expression-Recognition
   ```

2. დააყენეთ საჭირო პაკეტები:

   ```bash
   pip install -r requirements.txt
   ```

3. დააკავშირეთ Weights & Biases:

   ```bash
   wandb login
   ```

## 📁 პროექტის სტრუქტურა

```
Facial-Expression-Recognition/
├── data/                    # მონაცემების შესანახი საქაღალდე
├── models/                  # მოდელის არქიტექტურები
│   ├── cnn.py
│   ├── resnet.py
│   └── ...
├── notebooks/               # Jupyter ნოუთბუქები ექსპლორაციისთვის
├── src/                     # ძირითადი კოდი
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── configs/                 # კონფიგურაციის ფაილები
├── requirements.txt
└── README.md
```

## 🧠 მეთოდოლოგია

პროექტი მიჰყვება მოდელის განვითარებას ეტაპობრივად:

1. **საწყისი მოდელი**: მარტივი CNN არქიტექტურა
2. **მოწინავე კომპლექსურობა**: ეტაპობრივად ვზრდით არქიტექტურის სირთულეს
3. **რეგულარიზაცია**: სხვადასხვა ტექნიკა გადამეტებული მორგების თავიდან ასაცილებლად
4. **ტრანსფერული სწავლება**: წინასწარ გაწვრთნილი მოდელების გამოყენება
5. **ენისემბლი**: მრავალ მოდელის გაერთიანება უკეთესი შედეგისთვის

## 📈 ექსპერიმენტები და შედეგები

ყველა ექსპერიმენტი ლოგირდება Weights & Biases-ში. ლოგირებული მეტრიკები მოიცავს:

* სწავლებისა და ვალიდაციის დანაკარგი და სიზუსტე
* სასწავლო სიჩქარის ცვლადობა
* მოდელის ჰიპერპარამეტრები
* კონფუზიის მატრიცები
* ნიმუშების პროგნოზები


# Notebook 01_data_exploration.ipynb
## Data Exploration Results

### Dataset Overview
- **Size**: 28,709 training samples, 7,178 test samples
- **Classes**: 7 facial expressions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Image Format**: 48x48 grayscale images
- **Data Quality**: 0 missing values, 1236 duplicates

### Key Findings
1. **Class Imbalance**: Significant imbalance with ratio 16.55:1 (most:least common)
   - Most common: Happy 
   - Least common: Disgust

2. **Data Characteristics**:
   - Pixel values: 0-255 range
   - Mean pixel value: 131.0 ± 64.3
   - Consistent image dimensions

3. **Challenges Identified**:
   - Class imbalance will require balanced sampling or weighted loss
   - Low resolution (48x48) limits feature complexity
   - Grayscale only - no color information

### Data Split Strategy
- Training: 22,967 samples (80%)
- Validation: 5,742 samples (20%)  
- Stratified split to maintain class distribution

# Training
## 02_baseline_cnn.ipynb **Baseline CNN**

Simple Convolutional Neural Network for facial expression recognition. Serves as an initial model to establish performance baselines.

### 🔹 Version 1

#### 🏗 Architecture

```
Input (1, 48, 48)
├─ Conv2d(1, 32, kernel_size=5, padding=2)
├─ ReLU()
├─ MaxPool2d(kernel_size=2, stride=2)
├─ Conv2d(32, 64, kernel_size=5, padding=2)
├─ ReLU()
├─ MaxPool2d(kernel_size=2, stride=2)
├─ Flatten()
├─ Dropout(0.3)
├─ Linear(64 * 12 * 12, 128)
├─ ReLU()
├─ Dropout(0.3)
└─ Linear(128, 7)
```

#### ⚙️ Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Cross-entropy
- **Batch Size**: 64
- **Epochs**: 20 (early stopping)
- **Regularization**:
  - Dropout (0.3)

#### 📊 Results
- Training accuracy: ~85%
- Validation accuracy: ~60%
- **Issue**: Overfitting observed after 20 epochs


[Simple_cnn_v1](https://wandb.ai/ellekvirikashvili-free-university-of-tbilisi-/facial-expression-recognition/runs/70toflci?nw=nwuserellekvirikashvili)

---



### 🔹 Version 2 (Improved)

#### 🛠 Architecture Improvements
- **Batch Normalization** after each conv layer
- **Spatial Dropout (0.2)** in conv layers
- **Higher Dropout (0.6)** in FC layers
- **Reduced FC layers** (512 → 256 → 128)
- **Adaptive Pooling** for better input size handling

#### ⚙️ Training Configuration
- **Learning Rate**: 0.0005 (reduced from 0.001)
- **Weight Decay**: 1e-4 (L2 regularization)
- **Early Stopping** with patience=5
- **Learning Rate Scheduling**: Reduce on plateau
- **Batch Size**: 64 (unchanged)

#### 📊 Expected Improvements
- Better generalization
- Reduced overfitting
- More stable training

## 📁 `03_deeper_cnn.ipynb`

### 🧠 Deeper CNN with Batch Normalization

---

## 🔹 Version 1 (Deep\_CNN\_V1)

### 🏗 Architecture

* 7-შრიანი Convolutional ნერვული ქსელი (CNN)
* **4 Convolutional ბლოკი**, ყოველი დასრულებულია MaxPooling-ით
* ყოველი Conv-შრის შემდეგ გამოყენებულია **Batch Normalization**
* **Global Average Pooling** სრულდება FC ფენებამდე
* Dropout (0.5) რეგულარიზაციისთვის
* **Batch Normalization** გამოიყენება როგორც Convolutional, ისე Fully Connected ფენებში

### ⚙️ ჰიპერპარამეტრები

* Filters: `32 → 64 → 128 → 256`
* Optimizer: **Adam**, learning rate = `0.001`
* L2 weight decay: `1e-4`
* Epochs: `40` (Early stopping შესაძლებელი)
* Dropout rate: `0.5`
* Tracking: Weights & Biases ინტეგრაცია (`wandb`)

### 📉 Performance

* **Train Loss**: `0.1689`, **Train Accuracy**: `93.93%`
* **Val Loss**: `2.3061`, **Val Accuracy**: `57.72%`
* **Early Stopping**: განხორციელდა **24-ე ეპოქაზე**
* **Observation**: მოდელმა გადაჭარბებულად მოერგო ტრენინგ მონაცემებს — **overfitting**

📊 [Deep\_cnn\_v1 Run on W\&B](https://wandb.ai/ellekvirikashvili-free-university-of-tbilisi-/facial-expression-recognition/runs/f1pw8dnp?nw=nwuserellekvirikashvili)

---

## 🔹 Version 2 (Deep\_CNN\_V2)

### ⚙️ Key Changes from V1

* **Epochs შემცირდა**: `40 → 30`
* **Dropout გაიზარდა**: `0.5 → 0.7`
* **Spatial Dropout დამატებულია**:
  `self.dropout1 = nn.Dropout2d(0.1)` — ადრეულ ლეიერებს შორის, feature-level რეგულარიზაციისთვის
* **Channel კომპრესიები ბოლო Conv ფენებში**: `256 → 192`
* **FC ფენები გამარტივდა**: `512 → 256`
* **Early Stopping პარამეტრები გამკაცრდა**:

  ```python
  'early_stop_patience': 7,  # More aggressive early stopping
  'lr_patience': 3,          # Reduce LR sooner
  ```

### 🎯 Goal

* **Overfitting-ის შემცირება**
* **მოდელის გენერალიზაციის გაუმჯობესება**
* **მეტად სწრაფი და აგრესიული ადაპტაცია validation performance-ზე**

### ⏳ შედეგების მოლოდინი

* უკეთესი generalization-validation ბალანსი
* ნაკლები variance epochs-ს შორის
* ნაკლები training-validation gap


#### 📊 Results

* **Train Loss**: `0.1689`, **Train Accuracy**: `93.93%`
* **Val Loss**: `2.3061`, **Val Accuracy**: `57.72%`
* **Early Stopping**: განხორციელდა **24-ე ეპოქაზე**


[Deeper_cnn_v2](https://wandb.ai/ellekvirikashvili-free-university-of-tbilisi-/facial-expression-recognition/runs/ql1dpugq?nw=nwuserellekvirikashvili)


### V3

#### Basic Flow for Augmented Training:
`Tensor (from dataset) → ToPILImage() → PIL Augmentations → ToTensor() → Tensor Augmentations → Normalize`


#### Features:

±15° random rotation
50% horizontal flip probability
Random translation (±10% of image size)
Random scaling (90%-110%)
Random shear transformation
Brightness/contrast jitter
Random erasing (10% probability)
---

## ✅ დასკვნა

> პირველი ვერსია საკმარისად ძლიერი აღმოჩნდა training მონაცემებზე, მაგრამ ვერ გაართვა თავი validation-ზე — საჭირო გახდა აგრესიული რეგულარიზაცია და ადრეული learning rate decay.
> მეორე ვერსია მიდის **leaner architecture + smarter regularization** სტრატეგიით, რათა დაიბალანსოს სისწრაფე, სიზუსტე და სტაბილურობა.


## 04_attention_cnn.ipynb

### 🧠 Attention-based CNN Architecture

#### 🔍 Overview
A Convolutional Neural Network enhanced with Convolutional Block Attention Module (CBAM) that learns to focus on the most discriminative facial regions for expression recognition.

#### 🏗 Core Architecture

```
Input (1, 48, 48)
├─ Conv2d(1, 32) → BatchNorm → ReLU
├─ Conv2d(32, 32) → BatchNorm → ReLU → MaxPool2d(2)
└─ CBAM(32)  # First attention block

├─ Conv2d(32, 64) → BatchNorm → ReLU
├─ Conv2d(64, 64) → BatchNorm → ReLU → MaxPool2d(2)
└─ CBAM(64)  # Second attention block

├─ Conv2d(64, 128) → BatchNorm → ReLU
├─ Conv2d(128, 128) → BatchNorm → ReLU → MaxPool2d(2)
└─ CBAM(128)  # Third attention block

├─ AdaptiveAvgPool2d(1)
├─ Flatten
├─ Linear(128, 256) → BatchNorm → ReLU → Dropout(0.5)
└─ Linear(256, 7)  # 7 emotion classes
```

#### 🎯 Attention Mechanism (CBAM)

**Convolutional Block Attention Module** combines:

1. **Channel Attention**
   - Captures 'what' to focus on in the feature maps
   - Uses both average and max pooling paths
   - Learns channel-wise feature importance

2. **Spatial Attention**
   - Determines 'where' to focus in the spatial dimensions
   - Applies 1x1 convolutions to create spatial attention maps
   - Highlights important facial regions for expression recognition

#### ⚙️ Training Configuration
- **Optimizer**: AdamW with weight decay (1e-4)
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduling
- **Regularization**:
  - Dropout (0.5) in fully connected layers
  - L2 weight decay
  - Data augmentation (random horizontal flip, rotation)
- **Batch Size**: 64
- **Epochs**: 50 with early stopping

#### 📊 Performance Features
- **Visual Attention Maps**: Visualize which facial regions the model focuses on
- **Class Activation Mapping**: Understand model decisions
- **W&B Integration**: Track experiments and compare runs
- **Confusion Matrix**: Detailed performance analysis

#### 🚀 Key Benefits
1. **Improved Accuracy**: Focuses on relevant facial features
2. **Better Generalization**: Attention acts as a form of regularization
3. **Interpretability**: Visual explanations of model decisions
4. **Efficiency**: Lightweight attention modules with minimal computational overhead

#### 🛠 Implementation Details
- Uses PyTorch for model implementation
- Integrates with Weights & Biases for experiment tracking
- Includes comprehensive data augmentation
- Implements learning rate scheduling and early stopping

#### 📈 Expected Performance
- Training Accuracy: ~85-90%
- Validation Accuracy: ~60-65%
- Focuses on eyes, mouth, and eyebrow regions for expression recognition

## 🔍 Weights & Biases ინტეგრაცია

ყველა ექსპერიმენტი ლოგირდება Weights & Biases-ში შემდეგი სტრუქტურით:

* პროექტი: `facial-expression-recognition`
* ტეგები: `[model_type, dataset_version, experiment_type]`
* კონფიგი: ჰიპერპარამეტრები და არქიტექტურა
* მეტრიკები: სწავლების/ვალიდაციის შედეგები
* არტიფაქტები: მოდელის შენახული წონები

## 📝 მიგნებები და ანალიზი

ექსპერიმენტების ძირითადი მიგნებები:

1. **გადამეტებული მორგება**: მოგვარდა მონაცემთა აუგმენტაციითა და dropout-ით
2. **კლასების დისბალანსი**: გამოყენებულია წონიანი დანაკარგის ფუნქციები
3. **სასწავლო სიჩქარის ცვლა**: დიდი გავლენა აქვს კონვერგენციაზე
4. **მოდელის სიღრმე**: ბალანსი სირთულესა და ეფექტურობას შორის


