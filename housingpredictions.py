import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode

path = "/Users/adam/Desktop/DataSci/Portfolio-v1/housingdatamodeling/"



df = pd.read_csv(f"{path}datasets/train.csv")
#df.head()
#df.columns
df = df.dropna(axis=1, thresh=len(df)*.75)
#df.info()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr())

# Get dummies for categorical:

def add_dummies(col, df):
    dums = pd.get_dummies(df[col])
    dums = dums.iloc[:,1:]
    dums.columns = [f"{s}_{col.replace(' ','_')}" for s in dums.columns]
    return df.merge(dums, left_index=True, right_index=True)

df = add_dummies("Central Air", df)
# df.columns
plt.figure(figsize=(4,12))
sns.heatmap(pd.DataFrame(df.corr()["SalePrice"]).sort_values("SalePrice", ascending=False), annot=True)
plt.title("Correlation with Target")

# Look at outliers before imputing
#df["SalePrice"].plot.box()

# Create a list of columns to drop, and drop them.
# Utilities: no variance in column
# Land Slope: contour provides better prediction

drops = ["Id", "PID", "Utilities", "Land Slope", "Neighborhood", "Bldg Type", "House Style", "Exterior 1st", "Exterior 2nd", "Sale Type"]

df = df.drop(drops, axis = 1)

# Drop records with data errors
drop_records = [960] # Drop house 960 because it has 5x sqft, 2x room, etc. but price is $160,000. Appears to be data entry error...

df = df.drop(drop_records, axis=0) 
df.head()

# Look at the grouped mean and std of sale price within a feature
df.groupby(["Bsmt Exposure"]).SalePrice.agg(["mean","std"])


# Look at frequencies within categorical features
df["Bsmt Exposure"].value_counts()

# Great dummy variables for categorical features by selecting significant responses within the feature
def make_features(df):
    features = df.copy()

    features["Street"] = (features["Street"] == "Pave").astype(int)
    features["Lot Shape"] = features["Lot Shape"].apply(lambda x: ["Reg", "IR1", "IR2", "IR3"].index(x))
    features["Land Contour"] = (features["Land Contour"]=="HLS").astype(int)
    features["Lot Config"] = (features["Lot Config"] == "CulDSac").astype(int)
    features["Condition 1"] = (features["Condition 1"].isin(["PosA", "PosN"])).astype(int)
    features["Condition 2"] = (features["Condition 2"].isin(["PosA", "PosN"])).astype(int)
    features["Year Built"] = 2020 - features["Year Built"]
    features["Year Remod/Add"] = 2020 - features["Year Remod/Add"]
    features["Roof Style"] = features["Roof Style"].isin(["Shed", "Flat", "Hip"]).astype(int)
    features["Roof Matl"] = features["Roof Matl"].isin(["WdShngl", "WdShake"]).astype(int)

    features["Mas Vnr Type"] = features["Mas Vnr Type"].isin(["Stone", "BrkFace"]).astype(int)
    features = pd.get_dummies(features, columns= ["Exter Qual", "Exter Cond", "Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2", "Heating", "Heating QC", "Central Air", "Electrical", "Kitchen Qual", "Functional", "Garage Type", "Garage Finish", "Garage Qual", "Garage Cond", "Paved Drive", ], drop_first=True)
    features["Foundation"] = features["Foundation"].isin(["PConc"]).astype(int)

    features = features.drop(["Garage Cond_Po", "Garage Qual_Po"], axis=1)

    return features
    features.head().iloc[:,42:]


#---------- MODELING -------------
# Recursive Feature Elimination on Linear Regression... 
# Followed by cross-validated grid search for optimal parameters...

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


features = make_features(df)
df2 = features.loc[:,features.dtypes != object]

# Read in testing data to select features
df_test = pd.read_csv(f"{path}datasets/test.csv")

test_features = make_features(df_test)

# Drop features in testing data not present in training data
drop_features = test_features.columns.difference(df2.columns)
test_features = test_features.drop(drop_features, axis=1)
for col in test_features.columns:
    test_features.loc[test_features[col].isnull(), col] = test_features[col].median()

# Drop features in training data not present in testing data
drop_features = df2.columns.difference(test_features.columns)
y = df2.SalePrice
df2 = df2.drop(drop_features, axis=1)
df2["SalePrice"] = y


df2


y = df.SalePrice
X = df2
X = X.drop("SalePrice", axis=1)
for col in X.columns:
    X.loc[X[col].isnull(), col] = X[col].median()

# Scaling produces the same r2 for linear regression
# X = StandardScaler().fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=0)

reg = LinearRegression().fit(x_train,y_train)
reg.score(x_test,y_test)

sorted_by_coef = reg.coef_.argsort()

df2.iloc[:5, sorted_by_coef[::-1]]

# Plot predictions
plt.scatter(y_test, reg.predict(x_test))

# Plot residuals
plt.scatter(y_test, y_test-reg.predict(x_test))

#   Model 2: KNN

y = df.SalePrice
X = df2
X = X.drop("SalePrice", axis=1)
for col in X.columns:
    X.loc[X[col].isnull(), col] = X[col].median()


x_train, x_test, y_train, y_test = train_test_split(X,y, train_size=0.8, random_state=0)

for k in range(2,40):
    reg = KNN(k)
    reg.fit(x_train, y_train)
    print(k, reg.score(x_test, y_test))


    # Ridge and Lasso Models
from sklearn.linear_model import Ridge, Lasso

x = df2.drop(["SalePrice", "Misc Val"], axis=1).iloc[:, 2:]
for col in x.columns:
    x.loc[x[col].isnull(), col] = np.median(x[col].values)
    
y = df2.SalePrice

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

scaler = StandardScaler()
xtr_scaled = scaler.fit_transform(x_train)
xts_scaled = scaler.transform(x_test)

lasso = LassoCV(alphas=[0.1,0.2,0.5,0.8,1,2,5,9,18])
lasso.fit(xtr_scaled, y_train)
lasso.score(xts_scaled, y_test)


# Grid Search CV to find best parameters for lasso
from sklearn.model_selection import GridSearchCV

params = {
    "alpha": [.0001,1.4, 1.5, 1.6, 2.5, 10],
    "fit_intercept" : [True]
}

estimator = Lasso()

model = GridSearchCV(estimator, params)
model.fit(xtr_scaled, y_train)
model.score(xts_scaled, y_test)

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import RidgeCV

scaler = RobustScaler()
xtr_scaled = scaler.fit_transform(x_train)
xts_scaled = scaler.transform(x_test)

ridge = RidgeCV(alphas=[0.1,0.2,0.5,0.8,1,2,5,9,18])
ridge.fit(xtr_scaled, y_train)
ridge.score(xts_scaled, y_test)


# Grid Search CV to find best parameters for lasso
from sklearn.model_selection import GridSearchCV

params = {
    "alpha": [.01, .05, 1, 1.5, 2, 5],
    "fit_intercept" : [True]
}

estimator = Ridge()

model = GridSearchCV(estimator, params)
model.fit(xtr_scaled, y_train)
train_pred = model.predict(xts_scaled)
model.score(xts_scaled, y_test), model.best_params_


test_features_scaled = scaler.transform(test_features)
predictions = model.predict(test_features_scaled)
df_test["SalePrice"] = predictions
results = df_test[["Id", "SalePrice"]]
results = results.sort_values("Id").set_index("Id")

results.to_csv("sub_reg.csv")

sns.displot(results)

sns.displot(train_pred)


