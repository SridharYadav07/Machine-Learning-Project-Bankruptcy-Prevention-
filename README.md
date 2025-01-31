# Machine-Learning-Project-Bankruptcy-Prevention-
This project involves predicting whether a bank will experience bankruptcy or not using a set of financial features. The dataset contains information on factors like industrail risk, management risk, financiall flexibility, credibility, competitivness, and operating risk. The goal is to train multiple machine learning models to classify the bank into two categories: "bankruptcy"(0) and "non-bankruptcy"(1).

The model performance is evaluated based on:
-Accuracy
-Precision, Recall, and F1-score
-Confusion Matrix
-Regularization to avoid overfitting

The project also covers saving the best performing model using pickle for later use.
**Data Preprocessing:**
Data is cleaned, missing values are checked, and the target variables is redefined(bankruptcy = 0, non_bankruptcy =1).
Feature correlations are analyzed using heatmaps and pair plots.
**Model Building**
Logistic Regression with and without L1/L2 regularization (Lasso, Ridge).
Naive Bayes classifiers (GaussianNB and MultinomialNB).
Support Vector Machine (SVM) with different kernels: Linear, Polynomial, and Radial Basis Function.
**Evaluation**
Model performance is evaluated using accuracy scores, confusion matrices, and classification reports.
**Model Deployment**
The best performning SVM model (Polynomial kernel) is saved using pickle for future predictions.

Results:
The project demonstrates the effectiveness of different machine learning models for classifying bankruptcy.
The Logistic Regression model, after training on the full dataset, achieved an accuracy of 99.6%.
KNN achieved an accuracy of 98%, while Naive Bayes(both Gaussian and Multinomial) achieved 100% accuracy.
The best performing Support Vector Machine (SVM) model with a Polynomial kernel achieved an accuracy of 98%.
