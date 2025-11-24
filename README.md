# COS802_AfroXLMR_Project
# Overview
Problem: The AfriSenti dataset is a rich source of sentiment analysis data; however, it does not include isiXhosa or isiZulu, which are among the most widely spoken languages in South Africa and neighboring countries such as Mozambique, Zimbabwe, and Namibia. This creates a gap in representation and leads to low model performance due to the lack of exposure to these languages by pretrained multilingual models.  

To address this challenge, the project will apply transfer learning techniques, specifically cross-lingual transfer, by fine-tuning the multilingual model AfroXLMR on Xitsonga and Swahili, and then testing it on isiXhosa using zero-shot or few-shot setups. This approach enables evaluation of model performance in languages understood by target users, simulates AfriSenti-style sentiment analysis for isiXhosa using transfer learning and contributes to methodological innovation for low-resource African languages. The methodology is inspired by experiments conducted in the AfriSenti benchmark: A Twitter Sentiment Analysis Benchmark for African Languages.

## Summary Results
The notebook explores Afro-XLMR for sentiment analysis in low-resource African languages. The fine-tuned AfriSenti resulted in a weighted F1=0.88, but limited zero-shot transfer to isiXhosa with weighted F1=0.48. The few-shot adaptation (500 samples) improved results modestly (F1=0.53). This highlights need for balanced, culturally grounded data.

## Contents of Zip file
The zip file contains the notebook that executes the technical parts of the data science lifecycle(as the problem is defined in this readme.md file under 'Overview'),  starting with exploratory data analysis, cleaning the data, tokenizing the data, model training, model testing on isiXhosa and zero and few shot testing. The data is read in from online, the afrisenti dataset is read in directly from github and the data was mounted onto my university google drive. The isiXhosa corpus was read in directly through huggingface.

## Setup instructions

## Dataset and Data Description
Two datasets will be used, a primary and a supplementary source. The primary dataset is AfriSenti, which includes Swahili and Xitsonga, languages that are linguistically close to isiXhosa. The supplementary dataset is the Hugging Face isiXhosa Sentiment Corpus retrieved from michsethowusu/xhosa-sentiments-corpus · Datasets at Hugging Face, which contains sentiment-labeled text data in isiXhosa for binary classification (positive or negative). Sentiments were extracted and processed from English translations of the isiXhosa sentences using DistilBERT. The dataset is part of a broader collection of African language sentiment resources and contains 1,499,997 samples, with 841,785 (56.1%) labeled as positive and 658,212 (43.9%) as negative. Sentiment labels were generated using the distilbert-base-uncased-finetuned-sst-2-english model, with batch preprocessing optimized for efficiency. Duplicate entries were removed based on text content and only binary sentiment labels were retained.

