# 1. Import Packages
import pandas as pd
import numpy as np
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the Logistic Regression model and evaluation metrics from scikit-learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 2. Read Data
credit_df = pd.read_csv("./data/credit_train.csv", header = 0, sep = ',')

# 3. Data Processing & Cleansing
credit_df['Months since last delinquent'] = credit_df['Months since last delinquent'].fillna(0)

credit_df['Credit Score'] = credit_df[credit_df['Credit Score'] >= 900]['Credit Score'] / 10

# Fill missing credit score value by average of credit score
credit_df['Credit Score'] = credit_df['Credit Score'].fillna(round(credit_df['Credit Score'].mean()))

# Fill missing Annual Income by average of mean of Annual Income
credit_df['Annual Income'] = credit_df['Annual Income'].fillna(round(credit_df['Annual Income'].mean()))

credit_df['Years in current job'] = credit_df['Years in current job'].str.replace(' years', '').str.replace(' year', '').str.replace('< 1', '0.5').str.replace('+', '')
credit_df['Years in current job'] = credit_df['Years in current job'].astype('float')

credit_df.fillna({'Years in current job': credit_df['Years in current job'].median()}, inplace = True)

credit_df.dropna(inplace = True)

print(credit_df.head())
print(credit_df.describe())

# 4. Drop Columns - 'Loan ID' & 'Customer ID'
credit_df.drop(['Loan ID', 'Customer ID'], axis = 1, inplace=True)

# 5. Data Pre-Processing - Handling All Categorical Values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

binary_cols = [col for col in credit_df.columns if credit_df[col].dtype == 'object' and credit_df[col].nunique()>1]
print(binary_cols)

for col in binary_cols:
    # fit() - train label encoder with input samples
    # transform() - apply changes and transform existing dataframe samples.
    credit_df[col] = le.fit_transform(credit_df[col])

# 6. Features & Target
X = credit_df.drop('Loan Status', axis = 1)
Y = credit_df['Loan Status']

print(X.head())
print(Y.head())


# 1. Import Packages
import pandas as pd
import numpy as np
# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the Logistic Regression model and evaluation metrics from scikit-learn
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 2. Read Data
credit_df = pd.read_csv("./data/credit_train.csv", header = 0, sep = ',')

# 3. Data Processing & Cleansing
credit_df['Months since last delinquent'] = credit_df['Months since last delinquent'].fillna(0)

credit_df['Credit Score'] = credit_df[credit_df['Credit Score'] >= 900]['Credit Score'] / 10

# Fill missing credit score value by average of credit score
credit_df['Credit Score'] = credit_df['Credit Score'].fillna(round(credit_df['Credit Score'].mean()))

# Fill missing Annual Income by average of mean of Annual Income
credit_df['Annual Income'] = credit_df['Annual Income'].fillna(round(credit_df['Annual Income'].mean()))

credit_df['Years in current job'] = credit_df['Years in current job'].str.replace(' years', '').str.replace(' year', '').str.replace('< 1', '0.5').str.replace('+', '')
credit_df['Years in current job'] = credit_df['Years in current job'].astype('float')

credit_df.fillna({'Years in current job': credit_df['Years in current job'].median()}, inplace = True)

credit_df.dropna(inplace = True)

print(credit_df.head())
print(credit_df.describe())

# 4. Drop Columns - 'Loan ID' & 'Customer ID'
credit_df.drop(['Loan ID', 'Customer ID'], axis = 1, inplace=True)

# 5. Data Pre-Processing - Handling All Categorical Values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

binary_cols = [col for col in credit_df.columns if credit_df[col].dtype == 'object' and credit_df[col].nunique()>1]
print(binary_cols)

for col in binary_cols:
    # fit() - train label encoder with input samples
    # transform() - apply changes and transform existing dataframe samples.
    credit_df[col] = le.fit_transform(credit_df[col])

# 6. Features & Target
X = credit_df.drop('Loan Status', axis = 1)
Y = credit_df['Loan Status']

print(X.head())
print(Y.head())

# 7. Add SMOTE - Handling Imbalanced Dataset
from imblearn.over_sampling import SMOTE
smote = SMOTE()
transformed_feature, transformed_label = smote.fit_resample(X, Y)

# 9. Split Data into Train & Test
X_train, X_test, Y_train, Y_test = train_test_split(transformed_feature, transformed_label, test_size = 0.2,
                                                    random_state = 2)

# 10. Decision Tree Classifier
clf_tree_best = DecisionTreeClassifier(ccp_alpha = 0.001, criterion = 'gini', max_depth = 20, random_state = 2)

clf_tree_best.fit(X_train, Y_train)

joblib.dump(clf_tree_best, "./model/loan_default.pkl")

print("\nModel Saved Successfully!")

# save test sample as csv for prediction
X_test.to_csv("./data/x_test_sample.csv")
