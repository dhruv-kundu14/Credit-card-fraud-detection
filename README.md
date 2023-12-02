# Credit-card-fraud-detection

 

Problem Statement
The recent lockdown caused by the Covid outbreak, witnessed a sudden increase in online transactions. Due to extensive use of online payments, use of credit cards have increased at a rapid rate.  
●	This means there is more possibility of fraudulent transactions which eventually leads to heavy financial losses. 
●	Due to the large amount of data being processed every day, the model build is not fast enough to respond to scam in time.
●	Imbalanced Data i.e most of the transactions(99.8%) are not fraudulent which makes it really hard for detecting the fraudulent ones.
●	Due to advancement in technology, scammers are also using new methods to fraud people.
Therefore, banks and other financial institutions support the progress of credit card fraud detection applications.
Objectives
The main objective of the project is to detect fraud in credit card transactions. Other objectives includes -
●	The model build must be simple and fast enough to detect anomaly and classify it as fraud as soon as possible.
●	The model must reduce false positives as much as possible.
●	The model should reduce the number of verification measures and process data in real time.
●	For protecting the privacy of the user the dimensionality of the data can be reduced.
●	A more trustworthy source must be taken which double-checks the data, at least for training the model.
●	We can make the model simple and interpretable so that when the scammer adapts to it with just some tweaks we can have a new model up and running to deploy.
Introduction
In today's digital era credit cards have become very common. Reason being credit cards can be valuable tools for earning rewards, traveling, handling emergencies or unplanned expenses, and building credit. Many companies such as Slice, CRED, Uni-card have introduced many credit card schemes to lure consumers into using credit cards. Because of the increase in credit card users, the number of transactions have also increased at a rapid rate. For any bank or financial organization, credit card fraud detection is of utmost importance. Various frauds occurs due to -
●	Firstly and most ostensibly when your card details are overseen by some other person. 
●	When your card is lost or stolen and the person possessing it knows how to get things done. 
●	Fake phone call convincing you to share the details. 
●	And lastly and most importantly, a high-level hacking of the bank account details. 
The aim is, therefore, to create a classifier that indicates whether a requested transaction is a fraud or not. We overcome the problem by creating a binary classifier and experimenting with various machine learning techniques to see which fits better.
Credit Card Fraud Dataset
About the data: The data we are going to use is the Kaggle Credit Card Fraud Detection dataset. This dataset has 492 frauds out of 284,807 total transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. It contains features V1 to V28 which are the principal components obtained by PCA. We are going to neglect the ‘Time’ feature which is of no use to build the models. The remaining features are the ‘Amount’ feature that contains the total amount of money being transacted and the ‘Class’ feature that contains whether the transaction is a fraud case or not. ‘Class’ feature is the response variable and it takes value 1 in case of fraud and 0 otherwise.
Dataset - https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Methodology & Steps involved -
Importing dependencies -
 
Importing and displaying the data -
 
Output -
 
Histogram of the features -
Now we will be visualizing all the features from the dataset on graphs.
 
Output -
 
 
 
 
Data Processing and EDA -
Let’s have a look at how many fraud cases and valid cases are there in our dataset. Along with that, let’s also compute the percentage of fraud cases in the overall recorded transactions.
 
Output -
 
We can see that out of 284,807 samples, there are only 492 fraud cases which is only 0.17 percent of the total samples. So, we can say that the data we are dealing with is highly imbalanced data and needs to be handled carefully when modeling and evaluating.
Next, we are going to get a statistical view of both fraud and valid transaction amount data using the ‘describe’.
 
Output -	   As we can clearly notice from this, the average Money transaction for the fraudulent ones are more. This makes this problem crucial to deal with. Correlation matrix graphically gives us an idea of how features correlate with each other and can help us predict what are the features that are most relevant for the prediction.
 
 
 
 
In the HeatMap we can clearly see that most of the features do not correlate to other features but there are some features that either have a positive or a negative correlation with each other. For example “V2” and “V5” are highly negatively correlated with the feature called “Amount”. We also see some correlation with “V20” and “Amount”. 
Feature Selection & Data Split -
In this process, we are going to define the independent (X) and the dependent variables (Y). Using the defined variables, we will split the data into a training set and testing set which is further used for modeling and evaluating. We can split the data easily using the 
‘train_test_split’ algorithm.
 
Modeling # Building the Decision Tree & Random Forest Algorithm -
In this step, we will be building two different types of classification models namely Decision 
Tree & Random Forest. Even though there are many more models like K-Nearest Neighbors (KNN), Logistic Regression, Support Vector Machine (SVM), and XGBoost, which we can use, but for simplicity we used the above two for solving these  classification problems. Both these models can be built feasibly using the algorithms provided by the scikit-learn package.
 
Decision Tree algorithm is a supervised machine learning algorithm used for classification and regression tasks. Inside the algorithm, we have mentioned the ‘max_depth’ to be ‘4’ which means we are allowing the tree to split four times and the ‘criterion’ to be ‘entropy’ which is most similar to the ‘max_depth’ but determines when to stop splitting the tree. Finally, we have fitted and stored the predicted values into the ‘decision_tree_predict’ variable.
Random forest is a supervised machine learning algorithm. It creates a “forest” out of an ensemble of “decision trees”, which are normally trained using the “bagging” technique. The bagging method's basic principle is that combining different learning models improves the outcome.
We built the classifier using the ‘RandomForestClassifier’ algorithm and we mentioned the ‘max_depth’ to be 4 just like how we did to build the decision tree model. Finally, fitting and storing the values into the ‘random_forest_predict’.
Remember that the main difference between the decision tree and the random forest is that, the decision tree uses the entire dataset to construct a single model whereas, the random forest uses randomly selected features to construct multiple models. That’s the reason why the random forest model is used versus a decision tree.
Evaluation -
In this process we are going to evaluate our built models using the evaluation metrics provided by the scikit-learn package. Our main objective in this process is to find the best model for our given case. The evaluation metrics we are going to use are the accuracy score metric, precision score metric, recall score matric, f1 score metric, and matthews correlation coefficient metric.
 
 
 
 
Confusion Matrix -
Typically, a confusion matrix is a visualization of a classification model that shows how well the model has predicted the outcomes when compared to the original ones. Usually, the predicted outcomes are stored in a variable that is then converted into a correlation table. Using the correlation table, the confusion matrix is plotted in the form of a heatmap. Even though there are several built-in methods to visualize a confusion matrix, we are going to define and visualize it from scratch for better understanding.
 
 
 
Output Decision Tree Confusion Matrix	Random Forest Confusion Matrix
 
References
1.	Detecting Credit Card Fraud with Machine Learning, Aaron Rosenbaum, Stanford University, Stanford, CA, 94305, USA
2.	https://www.kaggle.com/mlg-ulb/creditcardfraud
3.	Credit Card Fraud Detection using Machine Learning Algorithms, Vaishnavi Nath Dornadulaa , Geetha Sa, VIT, Chennai, India 
THANK YOU
