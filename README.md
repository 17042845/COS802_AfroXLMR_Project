# COS802_AfroXLMR_Project
# 1. Overview
Problem: The AfriSenti dataset is a rich source of sentiment analysis data; however, it does not include isiXhosa or isiZulu, which are among the most widely spoken languages in South Africa and neighboring countries such as Mozambique, Zimbabwe, and Namibia. This creates a gap in representation and leads to low model performance due to the lack of exposure to these languages by pretrained multilingual models.  

To address this challenge, the project will apply transfer learning techniques, specifically cross-lingual transfer, by fine-tuning the multilingual model AfroXLMR on Xitsonga and Swahili, and then testing it on isiXhosa using zero-shot or few-shot setups. This approach enables evaluation of model performance in languages understood by target users, simulates AfriSenti-style sentiment analysis for isiXhosa using transfer learning and contributes to methodological innovation for low-resource African languages. The methodology is inspired by experiments conducted in the AfriSenti benchmark: A Twitter Sentiment Analysis Benchmark for African Languages.

## 2. Summary Results
The notebook explores Afro-XLMR for sentiment analysis in low-resource African languages. The fine-tuned AfriSenti resulted in a weighted F1=0.88, but limited zero-shot transfer to isiXhosa with weighted F1=0.48. The few-shot adaptation (500 samples) improved results modestly (F1=0.53). This highlights need for balanced, culturally grounded data.

## 4. Contents of Zip file
The zip file contains the notebook that executes the technical parts of the data science lifecycle(as the problem is defined in this readme.md file under 'Overview'),  starting with exploratory data analysis, cleaning the data, tokenizing the data, model training, model testing on isiXhosa and zero and few shot testing. The data is read in from online, the afrisenti dataset is read in directly from github and the data was mounted onto my university google drive. The isiXhosa corpus was read in directly through huggingface.

## 5. Setup instructions
### 5.1 Environment requirements
To ensure reproducibility, the experiments should be run using the following software versions:
- Python 3.10+
- PyTorch	2.x (with CUDA 11+ support for GPU training)
- Transformers (Hugging Face)	4.x
- Datasets 2.x
- Scikit-learn 1.x
- Pandas 2.x
- NumPy 1.26+

The packages are included in the ```requirements.txt``` file and are installed using ```pip install -r requirements.txt```

### 5.2 Hardware Requirements
|Stage|Hardware|Notes|
|-----|--------|-----|
|Data loading, EDA, preprocessing|CPU|Works on local machine|
|Model training / fine-tuning|GPU recommended|Experiments were conducted using Google Colab GPU (T4)|

Running training on CPU will work but will be significantly slower.

## 6. Running the code
All experiments are contained in the AFROXLMR_Experiment.ipynb. One can run the notebook on either Google Colab (recommended for GPU training) OR Local Jupyter Notebook
### 6.1 Enable GPU (Colab Only)
Go to: Runtime → Change runtime type → Hardware accelerator → GPU

### 6.2 Run All Cells
The notebook is organized into:
- Environment Setup
- Data Loading
- Exploratory Data Analysis (EDA)
- Preprocessing
- Tokenization
- Model Setup and Training
- Zero-shot Evaluation
- Few-shot Fine-tuning
- Results & Visualizations
  
No command-line arguments are required.
## Data information
Two datasets will be used, a primary and a supplementary source. The primary dataset is AfriSenti, which includes Swahili and Xitsonga, languages that are linguistically close to isiXhosa. The supplementary dataset is the Hugging Face isiXhosa Sentiment Corpus retrieved from michsethowusu/xhosa-sentiments-corpus· Datasets at Hugging Face, which contains sentiment-labeled text data in isiXhosa for binary classification (positive or negative). Sentiments were extracted and processed from English translations of the isiXhosa sentences using DistilBERT. The dataset is part of a broader collection of African language sentiment resources and contains 1,499,997 samples, with 841,785 (56.1%) labeled as positive and 658,212 (43.9%) as negative. Sentiment labels were generated using the distilbert-base-uncased-finetuned-sst-2-english model, with batch preprocessing optimized for efficiency. Duplicate entries were removed based on text content and only binary sentiment labels were retained.

## Citations
@inproceedings
{muhammadSemEval2023,
title = {{SemEval-2023 Task 12: Sentiment Analysis for African Languages (AfriSenti-SemEval)}},
author = {Shamsuddeen Hassan Muhammad and Idris Abdulmumin and Seid Muhie Yimam and David Ifeoluwa Adelani and Ibrahim Sa'id Ahmad and Nedjma Ousidhoum and Abinew Ali Ayele and Saif M. Mohammad and Meriem Beloucif and Sebastian Ruder},
booktitle = {Proceedings of the 17th {{International Workshop}} on {{Semantic Evaluation}} ({{SemEval-2023}})},
publisher = {{Association for Computational Linguistics}},
year = {2023},
url = {https://arxiv.org/pdf/2304.06845.pdf}
}

@dataset{xhosa_sentiments_corpus,
  title={Xhosa Sentiment Corpus},
  author={Mich-Seth Owusu},
  year={2025},
  url={https://huggingface.co/datasets/michsethowusu/xhosa-sentiments-corpus}
}


