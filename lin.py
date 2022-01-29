
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Load the Data
import zipfile
with zipfile.ZipFile("data/uber_lyft_ridesharing/rideshare_kaggle.zip", 'r') as zip_ref:
    zip_ref.extractall("data/uber_lyft_ridesharing/")


def load_data():
    csv_path = "data/uber_lyft_ridesharing/rideshare_kaggle.csv"
    return pd.read_csv(csv_path)


data = load_data()

# cleaning up useful attributes
df = data[["hour", "day", "month", "source", "destination", "cab_type",
           "product_id", "name", "price", "distance", "temperature", "short_summary",
           "long_summary", "humidity", "visibility", "uvIndex"]]

# select only uber data
df = df.loc[df['cab_type'] == "Uber"]

# df.describe()

df = df.dropna(subset=['price'])

# Finding Corelation
print(df.corr())

# Checking the unique data
for i in df.columns:
    if df[i].dtype == object:
        if len(df[i].unique()) < 1000:
            print("The unique values present in", i, "are: \n\n{}".
                  format(df[i].unique()), '\n')
        else:
            print("The unique values present in",
                  i, "are too large to display")


# df.head()

df = df.reset_index()

df["distance_range"] = pd.cut(df["distance"],
                              bins=[0., 1, 2, 3, 4, np.inf],
                              labels=[1, 2, 3, 4, 5])
# df["distance_range"].hist()
# plt.show()

# Spliting the Data in train and test Set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(df, df["distance_range"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

# strat_train_set.describe()
# strat_test_set.describe()


df_train = strat_train_set.drop("price", axis=1)
df_train_labels = strat_train_set["price"].copy()


num_attribs = ["hour", "day", "month", "distance",
               "temperature", "humidity", "visibility", "uvIndex"]
cat_attribs = [x for x in list(df_train) if (x not in num_attribs)]


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

df_train_prepared = full_pipeline.fit_transform(df_train)

# Liner Regeression
lin_reg = LinearRegression()
lin_reg.fit(df_train_prepared, df_train_labels)

# Testing on Some Data
some_data = df_train.iloc[:5]
some_labels = df_train_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

# Calculating RMSE
df_predictions = lin_reg.predict(df_train_prepared)
lin_mse = mean_squared_error(df_train_labels, df_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

# Cross Validation Score
print(np.sqrt(-cross_val_score(lin_reg, df_train_prepared,
                               df_train_labels, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)))

# lin_svm = SVR(kernel="poly", degree=3, C=100)
# lin_svm.fit(df_train_prepared, df_train_labels)

# some_data = df_train.iloc[:5]
# some_labels = df_train_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# print("Predictions:", lin_svm.predict(some_data_prepared))
# print("Labels:", list(some_labels))
# df_predictions = lin_svm.predict(df_train_prepared)
# svm_mse = mean_squared_error(df_train_labels, df_predictions)

# print(np.sqrt(svm_mse))
# print(np.sqrt(-cross_val_score(lin_svm, df_train_prepared,
#                                df_train_labels, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)))

# forest_reg = RandomForestRegressor(n_jobs=-1)
# forest_reg.fit(df_train_prepared, df_train_labels)

# some_data = df_train.iloc[:5]
# some_labels = df_train_labels.iloc[:5]

# some_data_prepared = full_pipeline.transform(some_data)

# print("Predictions:", forest_reg.predict(some_data_prepared))
# print("Labels:", list(some_labels))

# df_predictions = forest_reg.predict(df_train_prepared)
# lin_mse = mean_squared_error(df_train_labels, df_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print(forest_rmse)
# print(np.sqrt(-cross_val_score(forest_reg, df_train_prepared,
#                                df_train_labels, scoring="neg_mean_squared_error", cv=10, n_jobs=-1)))
