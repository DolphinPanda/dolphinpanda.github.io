---
layout: post
author: Seet Choon Wee
title: "Applied Data Science Project Documentation"
categories: ITD214
---
# Project Documentation

---

## Project Background

**Goal:**  
Improve online channel sales at Sephora.

**Objectives:**  
- Identify key factors driving product prices.  
- Predict customer sentiment towards beauty products based on review text.

Our project leverages data science to enhance Sephora’s online sales by providing actionable insights. By analyzing customer reviews, we aim to uncover the factors that influence product pricing and capture customer sentiment. These insights will help Sephora tailor its pricing strategies, product offerings, and marketing efforts to better meet customer needs and drive sales.

---

## Work Accomplished

### Data Preparation

#### Overview of the DataFrame
- Use `df.info()` to get an overview of the DataFrame:
  - Column names, data types, and the number of non-null values per column.

#### Check for Missing Values
- Use `df.isnull().sum()` to identify missing data in each column.
- Visual inspection with screenshots (e.g., via `df.info()` output).

![DataFrame Overview](https://github.com/user-attachments/assets/8544d4e0-a1b2-401d-98f6-4a057b2979ec)  
![Missing Values](https://github.com/user-attachments/assets/18b13e26-6d39-46be-8090-026d9786c7fc)

#### Handle Missing Values in `review_text`
- Remove rows with null values in the `review_text` column.
- Verify that all missing values are removed.

#### Data Cleaning: Checking for Duplicates
- **Objective:**  
  Ensure each review is unique to avoid bias or over-representation.
- **Steps Performed:**  
  - Identify duplicates using the `review_text` column.
  - Remove duplicate rows.
  - Verify the new dataset shape.
  
This step maintains data integrity for accurate analysis.

---

### Explore Rating Distribution

#### Overview
Understanding the distribution of ratings is crucial for predicting sentiment. Ratings serve as a numerical representation of customer sentiment—higher ratings indicate more positive feedback. Exploring the rating distribution reveals overall sentiment trends and potential data skewness.

#### Actions
- Visualize ratings with a histogram.
- Calculate summary statistics (mean, median, mode, standard deviation) to understand central tendency and spread.

![Rating Histogram](https://github.com/user-attachments/assets/10e89060-1254-462d-9b21-bbb9b59e1eb7)  
![Summary Statistics](https://github.com/user-attachments/assets/1b8a047c-76d0-46df-a5ef-aa00f91db27a)

---

### Visualizing Sentiment Distribution

- **Description:**  
  A pie chart visualizes the overall distribution of positive and negative sentiment.
  
- **Insights:**  
  - 17.9% of reviews are *negative*  
  - 82.1% of reviews are *positive*  
  This imbalance is a key consideration for modeling; future steps may involve resampling or adjusting class weights.

![Sentiment Pie Chart](https://github.com/user-attachments/assets/1688a92e-6f1c-4adb-9bf0-25e4d81ceb1b)

---

### Data Preprocessing: Tokenization, Stopwords Removal, and Stemming

- **Tokenization:**  
  The review text is split into individual words using NLTK's `word_tokenize`.

- **Stopwords Removal:**  
  Common English stopwords are removed using NLTK's stopword list, and additional common words (e.g., "use", "make", "one", "work", "day") are filtered out.

- **Punctuation and Numeric Removal:**  
  Only alphabetic tokens are retained, filtering out punctuation and numbers.

- **Stemming:**  
  The PorterStemmer reduces words to their root form for normalization.

The final processed text is stored in the `processed_review` column for further analysis.

---

### Word Frequency Distribution

- **Objective:**  
  Identify the most frequently occurring words in the processed reviews to understand common themes.

- **Steps:**  
  1. Combine all processed reviews into a single string.
  2. Use `Counter` to count word frequencies.
  3. Print the top 20 most common words.

Example Output:  
`[('skin', 1124023), ('use', 781944), ('product', 700713), ('love', 431849), ...]`

This analysis informs subsequent feature engineering decisions, such as removing non-informative words or incorporating n-grams.

![Top Words Bar Chart](https://github.com/user-attachments/assets/871a65da-4550-47ef-afda-9b2af5137892)

---

### Feature Extraction with TF-IDF

- **Objective:**  
  Convert the preprocessed review text into numerical features.

- **Process:**  
  - Split the data into training (80%) and testing (20%) sets.
  - Apply TF-IDF vectorization:
    - Limit features to 5000.
    - Capture unigrams and bigrams (`ngram_range=(1,2)`).
  - Transform the review text into TF-IDF matrices for model training and testing.

---

### Resampling the Training Data with Oversampling

- **Objective:**  
  Address class imbalance by oversampling the minority class (negative reviews).

- **Process:**  
  - Use `RandomOverSampler` on the TF-IDF training data to balance class distributions.
  - Verify the new class distribution to ensure balance.

![Oversampling Verification](https://github.com/user-attachments/assets/7e66dfe5-e70a-4af9-9967-bec78710682a)

---

### Model Comparison Using Confusion Matrix Analysis

**Objective:**  
Compare multiple classification models using not only accuracy but also confusion matrix metrics (precision, recall, F1-score) to understand their performance in detail.

**Approach:**  
- **Pipeline Construction:**  
  - For each model (Logistic Regression, Naive Bayes, Linear SVM, Random Forest), an imblearn pipeline is built that integrates:
    - TF-IDF vectorization (with `max_features=2000` and unigrams only)
    - Oversampling using `RandomOverSampler` to balance the dataset
    - The classifier under evaluation
- **Cross-Validation Predictions:**  
  - `cross_val_predict` is used with 2-fold cross-validation and parallel processing (`n_jobs=2`) to generate predictions for each model.
- **Performance Metrics:**  
  - For each model, the confusion matrix and classification report (including precision, recall, and F1-score) are computed.
  - These metrics provide a deeper understanding of model performance beyond simple accuracy.
  
**Outcome:**  
- The script prints the confusion matrix and classification report for each model, enabling a comprehensive comparison.
- The best model can then be identified based on a balanced evaluation of accuracy and detailed class-wise performance metrics.

---
### Optimized Pipeline Training Documentation

#### Overview
This optimized pipeline setup is designed to speed up the training process while maintaining robust model performance. The optimizations include:

- **Caching Intermediate Steps:**  
  Using the `memory` parameter in the pipeline to cache expensive transformations (e.g., TF-IDF vectorization) avoids recomputation during repeated training or cross-validation.

- **Reducing Feature Space:**  
  Lowering `max_features` in the `TfidfVectorizer` (from 2000 to 1500) reduces the dimensionality of the input features, leading to faster vectorization and model training.

- **Parallel Processing:**  
  Setting `n_jobs=-1` in the `RandomForestClassifier` leverages all available CPU cores to accelerate training.

- **Logging and Timing:**  
  Logging statements and timing measurements provide real-time feedback on each step of the process, making it easier to monitor and troubleshoot training performance.

#### Detailed Steps

1. **Cache Setup:**  
   - A temporary directory is created using `mkdtemp()` to store cached results.
   - The pipeline is configured with `memory=cachedir`, which caches the output of the `TfidfVectorizer` and any other expensive steps.

2. **Pipeline Configuration:**  
   - **TF-IDF Vectorizer:**  
     Configured with `max_features=1500` and `ngram_range=(1,1)` to reduce feature space.
   - **Oversampling:**  
     The `RandomOverSampler` balances class distribution.
   - **Classifier:**  
     The `RandomForestClassifier` is set with 100 trees, balanced class weights, and `n_jobs=-1` to use all CPU cores.

3. **Training & Logging:**  
   - The pipeline's training is timed using Python’s `time` module.
   - Logging messages indicate the start and completion of training, as well as any critical information such as the cache directory location.
   - The trained pipeline is verified by checking the classifier's attributes (e.g., `classes_`).

4. **Prediction:**  
   - Once the pipeline is trained, it can predict the sentiment of new reviews.
   - Logging messages are used to confirm the prediction process and display the predicted sentiments.

#### Business Value
- **Faster Iterations:**  
  Speeding up the training process allows for more rapid experimentation and model refinement.
- **Robust Model Deployment:**  
  The consistent and efficient pipeline ensures that new customer reviews are processed and predicted reliably, providing actionable insights to improve online sales.
- **Resource Efficiency:**  
  Optimized resource usage (memory and CPU) minimizes operational costs while maintaining high performance.

This documentation serves as a guide to understand the modifications made to optimize the training pipeline and explains the rationale behind each optimization step.

---
### Testing Multiple New Reviews Using the Persisted Pipeline

- **Objective:**  
  Evaluate the performance of the persisted pipeline by predicting sentiment for a set of new reviews, including two with a clearly positive tone and two with a clearly negative tone.

- **Process:**
  1. **Load the Pipeline:**  
     - The trained pipeline is loaded from disk using joblib's `load()` function.
  2. **Define New Reviews:**  
     - Four sample reviews are defined: two expected to be positive and two expected to be negative.
  3. **Prediction:**  
     - The loaded pipeline processes the new reviews (applying all necessary preprocessing and vectorization) and outputs predicted sentiments.
  4. **Output:**  
     - Each review along with its predicted sentiment is printed.

- **Business Value:**  
  - Demonstrates the model's capability to quickly and reliably classify new customer feedback.
  - Provides actionable insights into customer sentiment that can guide marketing strategies and product improvements to enhance online sales.
 
  ![image](https://github.com/user-attachments/assets/8ecb84a0-d935-4540-b72c-58284cee705c)
### Summary of New Review Predictions

- **Positive Reviews:**  
  - "I absolutely love this product! It has transformed my skin and left it glowing." → **Predicted Sentiment: positive**  
  - "This is the best skincare product I've ever used! My skin feels amazing." → **Predicted Sentiment: positive**

- **Negative Reviews:**  
  - "I hate this product. It ruined my skin and left it irritated." → **Predicted Sentiment: negative**  
  - "This product is awful. It made my skin break out and is completely ineffective." → **Predicted Sentiment: negative**

### Business Implications

- **Customer Feedback Analysis:**  
  The model accurately distinguishes between clearly positive and negative reviews. This can help Sephora:
  - **Leverage Positive Feedback:** Use positive sentiments in marketing campaigns to highlight popular products.
  - **Identify Areas for Improvement:** Investigate the root causes behind negative reviews to enhance product quality and customer satisfaction.

- **Actionable Insights:**  
  With reliable sentiment predictions, decision-makers can prioritize product enhancements, refine promotional strategies, and address customer concerns more effectively.

Overall, the successful predictions demonstrate the model's readiness for real-world applications, delivering valuable insights to support Sephora's business goal of improving online channel sales.
---

### Recommendation and Analysis

**Key Insights:**  
- Positive reviews emphasize benefits like improved skin feel and hydration.  
- Negative reviews highlight issues such as dryness, irritation, and usage confusion.

**Actionable Recommendations:**  
- Refine product messaging and usage instructions.  
- Develop targeted product lines for sensitive/dry skin.  
- Leverage positive feedback in marketing while addressing negative pain points.

**Business Impact:**  
- Enhanced customer satisfaction and targeted product improvements can drive higher online sales.

---
### Dashboard: Word Cloud for Positive Reviews

- **Purpose:**  
  Visualize the most frequently occurring words in positive reviews to understand what customers appreciate about the products.

- **Process:**  
  1. **Text Aggregation:**  
     - All reviews predicted as positive are concatenated into a single text string.
  2. **Word Cloud Generation:**  
     - The `WordCloud` library generates a visual representation, where word size indicates frequency.
  3. **Visualization:**  
     - The word cloud is displayed with no axis and a clear title.

- **Business Insight:**  
  - This visualization helps highlight key positive attributes mentioned by customers, which can be leveraged in marketing and product promotion to boost online sales.
 
![image](https://github.com/user-attachments/assets/43f5b000-4166-4bb6-a901-62e2ada8955f)

### Word Cloud of Positive Reviews: Observations and Next Steps

**Observations:**
- Common words include **“skin,” “feel,” “look,” “sensitive,” “dry,” “product”**—suggesting that many positive reviews focus on how the product impacts the user’s skin (e.g., making it feel better or look healthier).
- Words like **“leav,” “hydr,” “love”** also appear prominently, indicating satisfaction with moisturizing or hydrating effects.
- The emphasis on **“sensitive”** suggests that products which cater to sensitive skin may be particularly well-received.

**Potential Business Insights:**
1. **Highlight Product Benefits:**  
   - Marketing materials can emphasize the product’s ability to improve skin feel and appearance.
   - Specifically, mention benefits for those with **sensitive** or **dry** skin.

2. **Leverage Customer Language:**  
   - Use phrases customers commonly mention (e.g., “feel skin,” “make skin,” “look skin”) in campaigns to align marketing with customer perceptions.

3. **Address Negative Aspects:**  
   - If some words (e.g., “dry”) appear in a neutral or potentially negative context, investigate whether the product is meeting hydration expectations or if there’s confusion about product usage.

4. **Product Development Opportunities:**  
   - Consider developing or promoting lines specifically targeting **sensitive** or **dry** skin, as these are frequent themes in positive reviews.

**Next Steps:**
- Create a similar word cloud for **negative reviews** to compare and understand the pain points.
- Integrate additional visualizations (e.g., **bar charts** for the most common words, **time-series** plots if timestamps are available) to provide more nuanced insights.
- Use these findings to guide **marketing strategies**, **product enhancements**, and **customer support** initiatives, ultimately aiming to boost Sephora’s online sales.

This visualization of positive reviews helps Sephora understand the core attributes that customers value, informing data-driven decisions to enhance the customer experience and increase online channel sales.
---
### Dashboard: Word Cloud for Negative Reviews

- **Objective:**  
  Identify common words and themes in reviews predicted as negative, highlighting pain points and dissatisfaction areas.

- **Steps:**
  1. **Prediction Check:**  
     - If the `predicted_sentiment` column does not exist, generate predictions for a small sample and then for the full dataset.
  2. **Negative Text Aggregation:**  
     - Concatenate all reviews predicted as negative into a single string.
  3. **Word Cloud Generation:**  
     - Use the `WordCloud` library to visualize the frequency of words in negative reviews.
  4. **Visualization:**  
     - Display the word cloud without axis labels for clarity, and provide a descriptive title.

- **Business Value:**  
  - Understanding the language used in negative reviews helps Sephora pinpoint customer pain points, enabling targeted improvements to products, marketing strategies, and customer service efforts. 
  - Addressing these concerns can boost customer satisfaction and ultimately increase online sales.
 
![image](https://github.com/user-attachments/assets/5ef97ae8-77ca-425c-8d39-3d0a053e7543)
### Word Cloud of Negative Reviews: Observations and Recommendations

**Observations:**
- Words like **"skin," "dry," "sensit," "feel,"** and **"use product"** dominate, suggesting that many negative reviews still center around how the product impacts users’ skin.
- **"Realli," "want," "might"** appear as context words, which could indicate uncertainty or unmet expectations.

**Possible Reasons for Negative Sentiment:**
1. **Dry or Sensitive Skin Issues:**
   - Customers may feel the product is not sufficiently moisturizing or may be causing irritation on sensitive skin.
2. **Usage Confusion:**
   - Frequent references to “use product” might imply some confusion around how to apply the product effectively, leading to dissatisfaction.

**Business Implications:**
1. **Product Improvement:**
   - Investigate product formulas or instructions for use to ensure they address dryness or sensitivity adequately.
2. **Clearer Usage Guidelines:**
   - Enhance product labels, FAQs, and support materials to reduce confusion and improve results for customers.
3. **Targeted Support and Education:**
   - Provide more detailed guidance or tutorials on using the product to manage dryness or sensitivity issues.

**Next Steps:**
- **Compare Positive and Negative Clouds:**
  - Identify overlapping themes (e.g., "skin," "use product") and examine what differentiates positive experiences from negative ones.
- **Time-Series or Segment Analysis:**
  - If timestamps or product categories are available, segment negative reviews by date or product line to spot trends or isolated quality issues.
- **Actionable Follow-Up:**
  - Address the most common complaints through product modifications, clearer instructions, or targeted customer support to reduce negative sentiment and boost online sales.
---

### Dashboard: Top Words Bar Chart for Both Positive and Negative Reviews

- **Objective:**  
  Provide two separate bar charts showing the top words in positive reviews and the top words in negative reviews, respectively.

- **Steps:**
  1. **Define a Function (`plot_top_words`):**  
     - Takes a `sentiment_label` ("positive" or "negative") and the number of words to display (`num_words`).
     - Filters reviews based on the chosen sentiment, counts word frequency, and plots the top words in a horizontal bar chart.
  2. **Positive Reviews Chart:**  
     - Calls `plot_top_words('positive')` to display the top 20 words in positive reviews.
  3. **Negative Reviews Chart:**  
     - Calls `plot_top_words('negative')` to display the top 20 words in negative reviews.

- **Business Value:**  
  - Directly compares the most frequently used words in both positive and negative reviews, revealing key themes and pain points.
  - Aids decision-makers in addressing negative feedback and amplifying positive product attributes, ultimately driving better customer satisfaction and higher online sales.
![image](https://github.com/user-attachments/assets/d13d7cee-8396-4c8c-b6e8-55e8531d0218)
![image](https://github.com/user-attachments/assets/0a23f623-9c0f-4812-8018-67c6fe5ebb0f)
### Analysis of Top Words in Positive vs. Negative Reviews

**1. Common Themes Across Both Sentiments**  
- **"skin"** is the most frequently mentioned word in both positive and negative reviews, indicating that nearly all feedback (good or bad) revolves around the product’s impact on skin health or appearance.  
- **"use," "product," "feel," "like,"** and **"face"** also appear in both lists, suggesting that customers across the sentiment spectrum focus on product application, texture, and results.

**2. Differences in Word Frequency and Context**  
- **Positive Reviews:**  
  - Words like **"love," "great," "hydr," "moistur,"** and **"cream"** are more prominent, highlighting satisfaction with the product’s moisturizing and overall effectiveness.  
  - **"feel"** and **"like"** in a positive context likely reflect enjoyment or comfort in using the product.  
- **Negative Reviews:**  
  - **"product"** ranks higher than in positive reviews, often signifying direct dissatisfaction (e.g., “this product didn’t work”).  
  - Words like **"smell," "work," "get,"** and **"much"** appear, suggesting concerns about fragrance, effectiveness, or perceived value.  
  - **"dri"** (dry) is mentioned in both sentiments, but likely indicates a failure to hydrate in negative reviews versus successful hydration in positive ones.

**3. Potential Business Insights**  
- **Moisturizing vs. Dryness:**  
  - The recurring theme of dryness vs. hydration suggests that effectively addressing dryness or sensitivity is a crucial factor in customer satisfaction.  
  - Ensuring products deliver on moisturizing promises (and clarifying usage instructions) can reduce negative feedback.
- **Fragrance & Texture Concerns:**  
  - The prominence of **"smell"** in negative reviews indicates that fragrance (or odor) might be a frequent point of dissatisfaction.
- **Education & Usage Instructions:**  
  - Frequent references to **"use"** and **"product"** in negative reviews hint that customers may not be using the product correctly or have unmet expectations. Enhanced instructions, tutorials, or usage tips could help reduce negative experiences.

**4. Recommendations**  
1. **Focus on Dry/Sensitive Skin Solutions:**  
   - Given the emphasis on “dry” and “sensit,” expand product lines or provide clearer usage guidance for these concerns.
2. **Refine Product Messaging:**  
   - Highlight the benefits customers love in positive reviews (e.g., hydration, great feel) to attract new buyers.  
   - Provide disclaimers or instructions to manage expectations regarding fragrance or texture, especially if these elements are known to cause negative reactions.
3. **Enhance Customer Support & Education:**  
   - Offer more detailed guides or Q&A resources addressing common usage pitfalls, especially around dryness, sensitivity, and application techniques.
4. **Iterate & Monitor:**  
   - Regularly re-run this analysis on new reviews to track whether improvements (e.g., formula tweaks or better instructions) reduce negative feedback and bolster positive sentiment over time.

By examining the most frequent words in positive and negative reviews side by side, **Sephora** can pinpoint where products are excelling (hydration, feel) and where issues lie (dryness, fragrance, effectiveness), ultimately guiding data-driven decisions to improve online sales and customer satisfaction.
---
### Final Step: Consolidated Business Insights & Recommendations

After creating visualizations (pie charts, word clouds, bar charts) and analyzing positive vs. negative reviews, this final step translates your findings into clear, actionable insights for Sephora’s leadership. 

1. **Overall Sentiment Snapshot**  
   - A large proportion of reviews are positive, emphasizing benefits like hydration and improved skin feel.
   - Negative reviews often highlight dryness, sensitivity, or confusion about product usage.

2. **Key Positive Themes**  
   - **Moisturizing Benefits:** Words like “love,” “hydr,” “moistur” indicate strong approval of hydrating properties.  
   - **Skin Improvement:** Terms like “feel,” “look,” “great” underscore how customers appreciate visible improvements in their skin.

3. **Common Negative Pain Points**  
   - **Dry/Sensitive Skin:** Negative reviews frequently mention dryness or irritation, suggesting some customers experience adverse effects.  
   - **Fragrance Issues:** The word “smell” appears prominently, indicating dissatisfaction with scent for certain customers.  
   - **Usage Confusion:** References to “use product” in negative contexts may mean customers need more guidance.

4. **Actionable Recommendations**  
   - **Enhance Product Labeling & Usage Instructions:**  
     Provide clearer, more detailed instructions to mitigate dryness or irritation.  
   - **Focus on Sensitive Skin Solutions:**  
     Develop or promote specialized formulas for those with sensitive or dry skin.  
   - **Offer Fragrance-Free Options:**  
     Address common complaints about scent by offering milder or fragrance-free alternatives.  
   - **Continual Monitoring & Iteration:**  
     Re-run these analyses periodically to measure how product improvements and better instructions affect sentiment over time.

5. **Business Value**  
   - **Improved Customer Satisfaction:**  
     Addressing the issues highlighted in negative reviews can convert dissatisfied customers into repeat buyers.  
   - **Strategic Marketing & Product Development:**  
     Positive themes can be amplified in marketing, while negative themes guide product enhancements.  
   - **Increased Online Sales:**  
     By focusing on high-impact improvements (like dryness and fragrance concerns), Sephora can attract new customers and retain existing ones, driving revenue growth.

These consolidated insights serve as the final piece of your dashboard, ensuring that data-driven findings are communicated effectively to stakeholders and translated into meaningful business actions.
---
## AI Ethics

In developing our sentiment analysis model, we have carefully considered several key ethical principles to ensure our work is responsible, transparent, and fair. Below are the main areas of focus and how they are addressed in our project:

### Privacy
- **Justification:**  
  Customer reviews may include personal opinions and sensitive details. Protecting this information is critical for maintaining customer trust and complying with data protection regulations.
- **Application:**  
  - All customer data is anonymized, and any personally identifiable information (PII) is removed before analysis.
  - Data is securely stored and processed using industry best practices to prevent unauthorized access.

### Fairness
- **Justification:**  
  Bias in model predictions can lead to unfair treatment of specific groups and misinform business decisions, ultimately impacting customer satisfaction and market reputation.
- **Application:**  
  - We employ techniques like oversampling and class weighting to ensure that minority classes (e.g., negative reviews) are adequately represented during training.
  - The model is regularly evaluated for bias, and steps are taken to address any imbalances in sentiment predictions.
  - Our methodology is documented transparently to facilitate external audits and promote fairness.

### Accuracy
- **Justification:**  
  Reliable and accurate predictions are essential for effective decision-making. Inaccurate results can lead to misguided business strategies and a loss of customer trust.
- **Application:**  
  - The model undergoes rigorous evaluation using cross-validation and metrics such as accuracy, precision, recall, and F1-score.
  - Continuous monitoring and periodic retraining are planned to adapt to new data and maintain model performance.
  - Limitations and uncertainties in the model’s predictions are clearly communicated to stakeholders.

### Accountability
- **Justification:**  
  It is crucial to have clear accountability in AI systems so that decision-making processes are understood and responsibility is assigned for model outcomes.
- **Application:**  
  - Detailed documentation is maintained for all stages of the project, including data preprocessing, model training, and evaluation.
  - Our complete methodology and source code are available in our GitHub repository for auditability.
  - Clear roles and responsibilities have been defined among team members to address any issues during deployment or model updates.

### Transparency
- **Justification:**  
  Transparency in AI development fosters trust and enables stakeholders to understand how decisions are made, ensuring reproducibility and openness.
- **Application:**  
  - We provide comprehensive documentation that details our data sources, preprocessing steps, modeling decisions, and evaluation metrics.
  - Key decisions, such as the choice of algorithms and adjustments for class imbalance, are clearly explained.
  - Our project materials, including the code and reports, are publicly accessible, allowing for external review and feedback.

By addressing these ethical dimensions—privacy, fairness, accuracy, accountability, and transparency—we ensure that our sentiment analysis model is developed responsibly and provides reliable, actionable insights that align with both business goals and ethical standards.

---

## Source Codes and Datasets

https://github.com/DolphinPanda/itd214_proj
https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews/data 
