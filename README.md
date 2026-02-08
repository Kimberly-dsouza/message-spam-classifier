# Spam Detection Project

Short project: a supervised machine-learning pipeline to detect spam messages using the provided SMS dataset. The project includes exploratory data analysis, model training/validation in a notebook and a small inference app. The goal is to automatically classify emails as Spam or Not Spam (Ham) using Natural Language Processing (NLP) and Machine Learning techniques.

- Goal: Reduce spam delivery to users while minimizing false positives.
- Artifacts: [`spam_detection.ipynb`](spam_detection.ipynb) (EDA + modeling), [`app.py`](app.py) (inference/service), dataset: [`spam.csv`](spam.csv).
- Tech: Python, pandas, scikit-learn, Streamlit, Jupyter.

<img width="1365" height="608" alt="image" src="https://github.com/user-attachments/assets/1fa91e14-72f8-44e0-8e45-76c3dd2f7ed8" />


<img width="1365" height="575" alt="image" src="https://github.com/user-attachments/assets/d68803de-a409-4ad7-adc4-c1f5002ded75" />

## Approach & methodology
1. Data cleaning & EDA (class balance, token length, common tokens) — in [`spam_detection.ipynb`](spam_detection.ipynb).  
2. Feature engineering: text cleaning, TF-IDF / n-grams, basic lexical features.  
3. Model candidates: Logistic Regression, Naive Bayes, Random Forest, Voting Classifier, KNN, SVC. -- Naive Bayes model worked best with 0 FP.
4. Evaluation: stratified train/test split, cross-validation, precision/recall/F1, confusion matrix.  
5. Production: serialize model and serve via [`app.py`](app.py) or integrate into pipeline.
   - Use automated retraining schedule (weekly/monthly) if new spam patterns emerge.
   - Add human review queue for borderline predictions to reduce false positives.

## File structure
- [`app.py`](app.py) — inference/service code  
- [`spam_detection.ipynb`](spam_detection.ipynb) — full EDA and modeling workflow  
- [`spam.csv`](spam.csv) — dataset  
- [`LICENSE`](LICENSE)  
- [`.gitignore`](.gitignore)  
- [`README.md`](README.md) (this file)



## Business problem & impact
Detect spam messages automatically to:
- Improve end-user experience (fewer nuisance messages).
- Reduce platform abuse and associated costs.
- Maintain high true-positive detection while keeping false positives low to avoid blocking legitimate messages.

## Business KPIs to report:
- Messages flagged per 10k (volume).
- Expected customer complaints avoided per month.
- Operational cost saved per month from automated filtering.






