from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

bank_marketing = fetch_ucirepo(id=222)

X = bank_marketing.data.features
y = bank_marketing.data.targets
# Od razu dokonujemy podziału na train test val
X = X.sample(frac=1, random_state=42)  # tasujemy dane
y = y.loc[X.index] # dopasowujemy y do przetasowanego X

n = len(X)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

X_train = X.iloc[:train_end] # narazie reszty nie potrzebujemy, dorobimy później

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

def fillna_with_distribution(df, columns):
	for col in columns:		# rozkad liczymy na x_train
		value_counts = X_train[col].value_counts(normalize=True, dropna=True)
		missing = df[col].isna()
		if missing.sum() > 0:
			filled_values = np.random.choice(
				value_counts.index,
				size=missing.sum(),
				p=value_counts.values
			)
			df.loc[missing, col] = filled_values
	return df

X = fillna_with_distribution(X, ['job', 'education', 'contact'])

mask = (X['poutcome'].isna())
X.loc[mask, 'poutcome'] = 'none'

median_pdays = X_train.loc[X_train['pdays'].notna(), 'pdays'].median()
X.loc[X['pdays'].isna(), 'pdays'] = median_pdays.round(0)
X['pdays'] = X['pdays'].astype(int)
print(f"Mediana pdays z X_train: {median_pdays}")
print(X['pdays'].isna().sum())



# One-hot encoding 
dummy_cols = ['job', 'marital', 'poutcome']
X = pd.get_dummies(X, columns=dummy_cols, drop_first=False)

for col in ['default', 'housing', 'loan','contact']:
    if col == 'contact': 
        X[col] = X[col].map({'telephone': 1, 'cellular': 0})
    else:
        X[col] = X[col].map({'yes': 1, 'no': 0})


# label encoding dla education: 0 - primary, 1 - secondary, 2 - tertiary
enc_education = LabelEncoder()
X['education'] = enc_education.fit_transform(X['education'])

# cyclic encoding dla month
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
month_to_num = {month: idx for idx, month in enumerate(months)}
X['month_num'] = X['month'].map(month_to_num)
X['month_sin'] = np.sin(2 * np.pi * X['month_num'] / 12)
X['month_cos'] = np.cos(2 * np.pi * X['month_num'] / 12)

X = X.drop(columns=['month', 'month_num'])

X['day_sin'] = np.sin(2 * np.pi * X['day_of_week'] / 31)
X['day_cos'] = np.cos(2 * np.pi * X['day_of_week'] / 31)
X = X.drop(columns=['day_of_week'])
X.drop(columns=['duration','poutcome_none','is_first_time','marital_married','pdays'], inplace=True)  

q99 = X['campaign'].quantile(0.99)
X.loc[X['campaign'] > q99, 'campaign'] = q99
X.loc[X['balance'] < 0, 'balance'] = 0
X['balance'] = np.log1p(X['balance'])
X = X[X['previous'] < 100]
X['previous'] = np.log1p(X['previous'])

val_end = int(n * 0.9)

X_train = X.iloc[:train_end].copy()
X_val   = X.iloc[train_end:val_end].copy()
X_test  = X.iloc[val_end:].copy()

y['y'] = y['y'].map({'yes': 1, 'no': 0})
y_train = y.iloc[:train_end]
y_val   = y.iloc[train_end:val_end]
y_test  = y.iloc[val_end:]

scaler = StandardScaler()
cols_to_scale = ['age', 'balance', 'campaign', 'previous']

X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_val[cols_to_scale]   = scaler.transform(X_val[cols_to_scale])
X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

y_train_ = y_train.values.ravel().astype(int)
y_val_ = y_val.values.ravel().astype(int)
y_test_ = y_test.values.ravel().astype(int)

log_reg = LogisticRegression(max_iter=1000, random_state=42, penalty='l2', solver='lbfgs')
log_reg.fit(X_train, y_train_)

y_pred_log = log_reg.predict(X_val)