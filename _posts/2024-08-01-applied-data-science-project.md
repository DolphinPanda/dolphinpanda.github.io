---
layout: post
author: Seet Choon Wee
title: "Applied Data Science Project Documentation"
categories: ITD214
---
## Project Background

**Goal:**  
To improve online channel sales at Sephora.

**Objectives:**  
- Identify key factors driving product prices.  
- Predict customer sentiment towards beauty products based on review text.

Our project leverages data science to enhance Sephora’s online sales by providing actionable insights. By analyzing customer reviews, we aim to uncover the factors that influence product pricing and capture customer sentiment. These insights will help Sephora tailor its pricing strategies, product offerings, and marketing efforts to better meet customer needs and drive sales.

## Work Accomplished

### Data Preparation
#### Overview of the DataFrame
- Use `df.info()` to get an overview of the DataFrame:
  - Column names
  - Data types
  - Number of non-null values in each column

#### Check for Missing Values
- Use `df.isnull().sum()` to count the number of missing values in each column.
- This helps identify which columns have missing data and how many missing values they contain.

![image](https://github.com/user-attachments/assets/8544d4e0-a1b2-401d-98f6-4a057b2979ec)   
![image](https://github.com/user-attachments/assets/18b13e26-6d39-46be-8090-026d9786c7fc)

### Handle Missing Values in 'review_text'
#### Overview
The column `review_text` is critical for sentiment analysis as it contains the textual data required to determine customer sentiment. Since our analysis heavily relies on this text data, we must ensure there are no missing values. The approach we'll take is to remove any rows where the `review_text` is missing. This ensures that our dataset only contains complete reviews, which are valid for analysis.

#### Actions
- Remove rows with null values in the `review_text` column.
- Confirm that all missing `review_text` values have been successfully removed from the dataset.

### Data Cleaning: Checking for Duplicates

- **Objective:**  
  Ensure that each review in the dataset is unique to prevent any bias or over-representation in the analysis and model training.

- **Steps Performed:**  
  - **Identify Duplicates:**  
    - Checked for duplicate entries based on the `review_text` column.
  - **Remove Duplicates:**  
    - Dropped duplicate rows to maintain a dataset of unique review entries.
  - **Verification:**  
    - Confirmed the number of duplicates removed and verified the new shape of the dataset.

This step is crucial to maintain the integrity of the dataset and to ensure that subsequent preprocessing and modeling steps are based on unique and accurate data.

### Explore Rating Distribution
#### Overview
To predict customer sentiment, it's important to first understand the distribution of ratings given by customers. Ratings are a numerical representation of sentiment, where higher ratings typically indicate more positive sentiment. By exploring the rating distribution, we can gain insights into the overall sentiment towards the beauty products and identify any potential skewness or biases in the data.

#### Actions
- Visualize the distribution of ratings using a histogram.
- Calculate summary statistics for the ratings to understand central tendency and spread.

![image](https://github.com/user-attachments/assets/10e89060-1254-462d-9b21-bbb9b59e1eb7)
![image](https://github.com/user-attachments/assets/1b8a047c-76d0-46df-a5ef-aa00f91db27a)

### Visualizing Sentiment Distribution

In this step, we create a pie chart to visualize the distribution of positive and negative sentiment labels in our dataset. This visualization provides a clear overview of the balance between the two sentiment classes, helping us identify any potential class imbalances that might affect our model's performance.

The pie chart is generated using the `matplotlib` library. We calculate the count of each sentiment label from our cleaned DataFrame, then plot these values with distinct colors and percentage labels for better interpretability.

![image](https://github.com/user-attachments/assets/1688a92e-6f1c-4adb-9bf0-25e4d81ceb1b)

### Sentiment Distribution Analysis

I generated a pie chart to visualize the sentiment distribution in the dataset. The chart shows that **17.9%** of the reviews are *negative* and **82.1%** are *positive*. This significant imbalance in sentiment distribution is an important consideration for the modeling process, as it could influence model performance. Future steps may involve addressing this imbalance through methods such as resampling or adjusting class weights.

### Data Preprocessing: Tokenization, Stopwords Removal, and Stemming

- **Tokenization:** 
  - The review text is tokenized using NLTK's `word_tokenize` to split the text into individual words.
  
- **Stopwords Removal:** 
  - Common English stopwords are removed using NLTK's stopword list to focus on informative words.
  
- **Punctuation and Numeric Removal:** 
  - Only alphabetic tokens are kept, filtering out punctuation and numbers.
  
- **Stemming:** 
  - The PorterStemmer is applied to reduce words to their root form, helping to normalize the text.
  
The processed review text is stored in the new column `processed_review` for subsequent analysis.

### Word Frequency Distribution

- **Objective:**  
  Identify the most frequently occurring words in the processed reviews to gain insights into common themes or terms used by customers.

- **Steps Performed:**
  1. Combined all processed reviews into a single string.
  2. Used `Counter` to count the frequency of each word.
  3. Printed the top 20 most common words.

This frequency distribution helps in understanding the dominant words in the dataset, which can influence feature engineering decisions. For instance, one may decide to remove overly common words that do not carry much meaning (e.g., domain-specific stopwords) or explore adding bigrams/trigrams for deeper context.

![image](https://github.com/user-attachments/assets/871a65da-4550-47ef-afda-9b2af5137892)

### Feature Extraction with TF-IDF

- **Objective:**  
  Convert the preprocessed review text into numerical features using TF-IDF.

- **Process:**
  - **Data Splitting:**  
    - The dataset is divided into training (80%) and testing (20%) sets.
  - **TF-IDF Vectorization:**  
    - The `TfidfVectorizer` is configured to:
      - Limit the feature space to a maximum of 5000 features.
      - Capture both unigrams and bigrams (`ngram_range=(1,2)`).
  - **Result:**  
    - The review text is transformed into TF-IDF matrices for both training and testing sets.


### Resampling the Training Data with Oversampling

- **Objective:**  
  Address the class imbalance by oversampling the minority class (negative reviews) in the training data.

- **Process:**
  - **Oversampling:**  
    - The `RandomOverSampler` from the `imblearn` package is applied to the TF-IDF features (`X_train_tfidf`) and the corresponding labels (`y_train`).
  - **Result:**  
    - The oversampling creates a balanced training set, ensuring that both negative and positive classes are equally represented.
  - **Verification:**  
    - The new class distribution is printed to confirm that oversampling has balanced the training data.

This step ensures that the model training is based on a balanced dataset, which can help improve the performance on the minority class.

    
  ![image](https://github.com/user-attachments/assets/7e66dfe5-e70a-4af9-9967-bec78710682a)

### Modelling
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

### Evaluation
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Recommendation and Analysis
Explain the analysis and recommendations

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## AI Ethics
Discuss the potential data science ethics issues (privacy, fairness, accuracy, accountability, transparency) in your project. 

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt. Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum. In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris. Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

## Source Codes and Datasets
Upload your model files and dataset into a GitHub repo and add the link here. 
https://github.com/DolphinPanda/itd214_proj
