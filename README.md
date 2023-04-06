# Suicidal Text Detection

This project aims to build a predictive model to detect suicidal intent in social media posts, and to integrate the model into a functional mental health chatbot.

## Project Resources
For more project details, please refer to the following resources linked here:
- [Project Report](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/gohjiayi/suicidal-text-detection/master/docs/Suicidal-Text-Detection_Report.pdf)
- [Project Slides](https://docs.google.com/viewer?url=https://raw.githubusercontent.com/gohjiayi/suicidal-text-detection/master/docs/Suicidal-Text-Detection_Slides.pdf)
- [Project Video](https://youtu.be/RpvtJ32cqqA)

If you are unable to view the report and slides, navigate to the `docs/` folder.

## Project Directory Structure
```
├── data_preprocessing.ipynb
├── data_cleaning.ipynb
├── eda.ipynb
├── word2vec.ipynb
├── models_logit.ipynb
├── models_cnn.ipynb
├── models_lstm.ipynb
├── models_bert.ipynb
├── models_electra.ipynb
├── infer.ipynb
├── chatbot.ipynb
├── Data
│   ├── suicide_detection.csv
│   ├── suicide_detection_full_clean.csv
│   ├── suicide_detection_final_clean.csv
│   └── ...
├── Models
│   └── ...
└── …
```
This project is built on Python 3 and scripts were originally hosted on Google Colab. Required packages are installed individually in each `.ipynb` file.

The [`Data/`](https://drive.google.com/drive/folders/15wje4eEGWjxq15KlVl8EuDiq6bPlpobi?usp=sharing) folder consists of the dataset and embeddings used, while the [`Models/`](https://drive.google.com/drive/folders/1QH5wrcaaIBMM71Ozw53CyVd84zOWFlEQ?usp=sharing) folder consists of the trained models.

## 1. Data Collection
The [Suicide and Depression Detection dataset](https://www.kaggle.com/nikhileswarkomati/suicide-watch) is obtained from Kaggle and stored as `Data/suicide_detection.csv`. The dataset was scraped from Reddit and consists of 232,074 rows equally distributed between 2 classes - suicide and non-suicide.

## 2. Text Preprocessing
Run `data_preprocessing.ipynb` to perform text preprocessing and generate `Data/suicide_detection_full_clean.csv`.
> Note: Spelling correction requires a long processing time.

## 3. Data Cleaning
Run `data_cleaning.ipynb` to clean data and generate `Data/suicide_detection_final_clean.csv`. This is the final dataset which will be used for the project and will be split into train:test:val sets with the ratio of 8:1:1. The final dataset is left with 174,436 rows with a class distribution of approximately 4:6 for suicide and non-suicide.

## 4. Exploratory Data Analysis (EDA)
Run `eda.ipynb` to gain more insights of both the original dataset and the cleaned dataset. EDA on the original dataset helped to determine steps for text preprocessing and data cleaning, while insights gained on the cleaned train data helped us to better build our representations and models.

## 5. Representation Learning
Run `word2vec.ipynb` to pre-train custom Word2Vec embeddings from our cleaned dataset and to generate `Data/vocab.txt` and `Data/embedding_word2vec.txt`. In our project, we have also experimented with pre-trained Twitter GloVe embeddings retrieved [here](https://nlp.stanford.edu/projects/glove/), which was downloaded and stored as `Data/glove_twitter_27B_200d.txt`.

## 6. Model Building & Evaluation
There are 5 models built for this project - Logistic Regression (Logit), Convolutional Neural Network (CNN), Long Short-term Memory Network (LSTM), BERT, and ELECTRA. These models are stored separately into different notebooks with the format `models_[model name].ipynb`. The trained models are stored in the `Models/` folder.
> Note: Different variations were built for each model to find the best hyperparameters by testing empirically.

## 7. Model Selection
The best variation of the aforementioned models can be seen in the table below. Although BERT and ELECTRA have rather comparable performances, **ELECTRA** is selected as the best performing model with our prioritisation on F1 score, as well as insights into the model architecture. Run `infer.ipynb` to predict whether input text has suicidal intent or not using the selected ELECTRA model.

| Best Model | Accuracy | Recall | Precision | F1 Score |
|:---:|:---:|:---:|:---:|:---:|
| Logit | 0.9111 | 0.8870 | 0.8832 | 0.8851 |
| CNN | 0.9285 | 0.9013 | 0.9125 | 0.9069 |
| LSTM | 0.9260 | 0.8649 | 0.9386 | 0.9003 |
| BERT | 0.9757 | 0.9669 | **0.9701** | 0.9685 |
| **ELECTRA** | **0.9792** | **0.9788** | 0.9677 | **0.9732** |

The suicidal BERT and ELECTRA text classification models trained are available on HuggingFace at [gooohjy/suicidal-bert](https://huggingface.co/gooohjy/suicidal-bert) and [gooohjy/suicidal-electra](https://huggingface.co/gooohjy/suicidal-electra).

## 8. Chatbot Integration
Run `chatbot.ipynb` to use the mental health chatbot, integrated with the suicidal detection model. The chatbot is based on DialoGPT and custom retrieval-based responses were integrated to suit our use case.

## Contributors
- Aeron Sim Shih Wen - [@aeronsimshihwin](https://github.com/aeronsimshihwin)
- Goh Jia Yi - [@gohjiayi](https://github.com/gohjiayi)
- Lim Zi Hui - [@limzihui](https://github.com/limzihui)
- Lin Xiao - [@anqier98](https://github.com/anqier98)
- Quek Yi Zhen - [@quekyizhen](https://github.com/quekyizhen)
