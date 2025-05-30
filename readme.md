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

### გამოყენებული არქიტექტურები

1. **მარტივი CNN**

   * 3-4 კონვოლუციური ფენა
   * MaxPooling და BatchNorm
   * Dropout რეგულარიზაციისთვის

2. **ResNet ვარიანტები**

   * ResNet18
   * ResNet34
   * წონებით ან წონების გარეშე

3. **EfficientNet**

   * სხვადასხვა მასშტაბირების ვარიანტები
   * Fine-tuning სტრატეგიები

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

## 🚀 გამოყენება

### მოდელის გაწვრთნა

```bash
python src/train.py --model cnn --epochs 50 --batch_size 64 --lr 0.001
```

### შეფასება

```bash
python src/evaluate.py --model_path models/best_model.pth
```

## 🤝 შეუერთდი პროექტს

მოგვაწოდე შენი წვლილი! მოხარული ვიქნებით Pull Request-ების მისაღებად.

## 📄 ლიცენზია

ეს პროექტი ლიცენზირებულია MIT ლიცენზიით — დეტალებისთვის იხილეთ ფაილი [LICENSE](LICENSE).


