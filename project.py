# %%


# %%streamlit run yourscript.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor ## knn
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

# %%
df = pd.read_csv("environment_air_quality.csv")
df.head()

# %%
df= df.drop(columns="station_id")

# %%
df.head()

# %%
df.columns

# %%
df.describe()

# %%
df.info()

# %%
df.columns

# %% [markdown]
# ##Missing values

# %%
df.isnull().sum()

# %%
df.dropna().shape

# %%
import scipy as sc

# %%
sc.stats.jarque_bera(df["no2_ppb"].dropna())

# %%
df["no2_ppb"] = df["no2_ppb"].fillna(np.mean(df["no2_ppb"].dropna()))

# %%
df.isnull().sum()

# %%
sc.stats.jarque_bera(df["humidity_pct"].dropna())

# %%
df["humidity_pct"] = df["humidity_pct"].fillna(np.mean(df["humidity_pct"].dropna()))

# %%
df.isnull().sum()

# %%
sc.stats.jarque_bera(df["green_cover_pct"].dropna())

# %%
df["green_cover_pct"] = df["green_cover_pct"].fillna(np.median(df["green_cover_pct"].dropna()))

# %%
df.isnull().sum()

# %% [markdown]
# ## Outeliers

# %%
df.columns

# %% [markdown]
# # Box plot

# %%
fig, axes = plt.subplots(3,3, figsize = (16,8))
sns.boxplot(data=df, x= "traffic_volume_k", ax = axes[0,0], color = "red")
axes[0,0].set_title("Boxplot of traffic_volume")
sns.boxplot(data=df, x= "no2_ppb", ax = axes[0,1], color = "yellow")
axes[0,1].set_title("Boxplot of no2_ppb ")
sns.boxplot(data=df, x= "population_density", ax = axes[0,2])
axes[0,2].set_title("box plot of population_density")
sns.boxplot(data=df, x= "industrial_index", ax = axes[1,0], color = "brown")
axes[1,0].set_title("Box plot ofindustrial_index")
sns.boxplot(data=df, x= "wind_speed_ms", ax = axes[1,1])
axes[1,1].set_title("Box plot of wind_speed_ms")
sns.boxplot(data=df, x= "humidity_pct", ax = axes[1,2], color= "gray")
axes[1,2].set_title("Box plot of humidity_pct ")
sns.boxplot(data=df, x= "green_cover_pct", ax = axes[2,0])
axes[2,0].set_title("Box plot of green_cover_pct")
sns.boxplot(data=df, x= "temp_c", ax = axes[2,1], color= "green")
axes[2,1].set_title("Box plot of temp_c")
sns.boxplot(data=df, x= "pm25_ugm3", ax = axes[2,2], color= "green")
axes[2,2].set_title("Box plot of pm25_ugm3 ")
plt.suptitle("Boxplot of All Variables", fontsize = 20 )
plt.tight_layout()
plt.show()



# %%
df.columns

# %%
Q1 = df["traffic_volume_k"].quantile(0.25) 
Q3 = df["traffic_volume_k"].quantile(0.75)
IQR = Q3 - Q1
lower_band = Q1 - IQR*1.5
upper_band = Q3 + IQR*1.5
df["traffic_volume_k"] = np.clip(df["traffic_volume_k"],lower_band, upper_band)


# %%
Q1 = df["no2_ppb"].quantile(0.25) 
Q3 = df["no2_ppb"].quantile(0.75)
IQR = Q3 - Q1
lower_band = Q1 - IQR*1.5
upper_band = Q3 + IQR*1.5
df["no2_ppb"] = np.clip(df["no2_ppb"],lower_band, upper_band)

# %%
Q1 = df["population_density"].quantile(0.25) 
Q3 = df["population_density"].quantile(0.75)
IQR = Q3 - Q1
lower_band = Q1 - IQR*1.5
upper_band = Q3 + IQR*1.5
df["population_density"] = np.clip(df["population_density"],lower_band, upper_band)

# %%
Q1 = df["industrial_index"].quantile(0.25) 
Q3 = df["industrial_index"].quantile(0.75)
IQR = Q3 - Q1
lower_band = Q1 - IQR*1.5
upper_band = Q3 + IQR*1.5
df["industrial_index"] = np.clip(df["industrial_index"],lower_band, upper_band)

# %%
Q1 = df["wind_speed_ms"].quantile(0.25) 
Q3 = df["wind_speed_ms"].quantile(0.75)
IQR = Q3 - Q1
lower_band = Q1 - IQR*1.5
upper_band = Q3 + IQR*1.5
df["wind_speed_ms"] = np.clip(df["wind_speed_ms"],lower_band, upper_band)

# %%
Q1 = df["humidity_pct"].quantile(0.25) 
Q3 = df["humidity_pct"].quantile(0.75)
IQR = Q3 - Q1
lower_band = Q1 - IQR*1.5
upper_band = Q3 + IQR*1.5
df["humidity_pct"] = np.clip(df["humidity_pct"],lower_band, upper_band)

# %%
Q1 = df["green_cover_pct"].quantile(0.25) 
Q3 = df["green_cover_pct"].quantile(0.75)
IQR = Q3 - Q1
lower_band = Q1 - IQR*1.5
upper_band = Q3 + IQR*1.5
df["green_cover_pct"] = np.clip(df["green_cover_pct"],lower_band, upper_band)

# %%
Q1 = df["temp_c"].quantile(0.25) 
Q3 = df["temp_c"].quantile(0.75)
IQR = Q3 - Q1
lower_band = Q1 - IQR*1.5
upper_band = Q3 + IQR*1.5
df["temp_c"] = np.clip(df["temp_c"],lower_band, upper_band)

# %%
Q1 = df["pm25_ugm3"].quantile(0.25) 
Q3 = df["pm25_ugm3"].quantile(0.75)
IQR = Q3 - Q1
lower_band = Q1 - IQR*1.5
upper_band = Q3 + IQR*1.5
df["pm25_ugm3"] = np.clip(df["pm25_ugm3"],lower_band, upper_band)

# %%
fig, axes = plt.subplots(3,3, figsize = (16,8))
sns.boxplot(data=df, x= "traffic_volume_k", ax = axes[0,0], color = "red")
axes[0,0].set_title("Boxplot of traffic_volume")
sns.boxplot(data=df, x= "no2_ppb", ax = axes[0,1], color = "yellow")
axes[0,1].set_title("Boxplot of no2_ppb ")
sns.boxplot(data=df, x= "population_density", ax = axes[0,2])
axes[0,2].set_title("box plot of population_density")
sns.boxplot(data=df, x= "industrial_index", ax = axes[1,0], color = "brown")
axes[1,0].set_title("Box plot ofindustrial_index")
sns.boxplot(data=df, x= "wind_speed_ms", ax = axes[1,1])
axes[1,1].set_title("Box plot of wind_speed_ms")
sns.boxplot(data=df, x= "humidity_pct", ax = axes[1,2], color= "gray")
axes[1,2].set_title("Box plot of humidity_pct ")
sns.boxplot(data=df, x= "green_cover_pct", ax = axes[2,0])
axes[2,0].set_title("Box plot of green_cover_pct")
sns.boxplot(data=df, x= "temp_c", ax = axes[2,1], color= "green")
axes[2,1].set_title("Box plot of temp_c")
sns.boxplot(data=df, x= "pm25_ugm3", ax = axes[2,2], color= "green")
axes[2,2].set_title("Box plot of pm25_ugm3 ")
plt.suptitle("Boxplot of All Variables", fontsize = 20 )
plt.tight_layout()
plt.show()



# %% [markdown]
# ## Regression

# %%
## EDA
df.head()

# %%
df.describe()

# %%
df.info()

# %% [markdown]
# ##features and target

# %%
X = df.iloc[:,:-1] 
y = df.iloc[:,-1]

# %% [markdown]
# ## Correlation

# %%
cor = X.corr()
sns.heatmap(cor, annot= True, fmt = ".2f", cmap = "viridis")

# %% [markdown]
# # split and scaling

# %%
#Train and Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42 )
#Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
X_train.head()

# %% [markdown]
# ## Last squar

# %%
model = LinearRegression()
model_ls = model.fit(X_train, y_train)

# %% [markdown]
# ## Gardient

# %%
sgd_model = SGDRegressor(max_iter=1000, random_state=42)
model_sgd = sgd_model.fit(X_train_scaled, y_train)

# %% [markdown]
# ## KNN

# %%
model = KNeighborsRegressor(n_neighbors=20)
model_KNN = model.fit(X_train_scaled, y_train)

# %% [markdown]
# ## Ridge

# %%
model = Ridge(alpha=0.05)
model_ridge = model.fit(X_train_scaled, y_train)

# %% [markdown]
# ## Lasso

# %%
model = Lasso(alpha=0.05)
model_lasso = model.fit(X_train_scaled, y_train)

# %% [markdown]
# ## Compare Models

# %%
# Assuming your models and names lists are already defined
models = [model_ls, model_sgd, model_KNN, model_ridge, model_lasso]
names = ["Least Squares", "Gradient Descent","K-Nearest Neighbors","Ridge Regression","Lasso Regression"]
results = []
# Loop through each model to calculate metrics
for name, model in zip(names, models):
    # Route the unscaled data to Least Squares, scaled data to the rest
    if name == "Least Squares":
        preds = model.predict(X_test)
    else:
        preds = model.predict(X_test_scaled)
    results.append(
        {"Model": name,"MSE": round(mean_squared_error(y_test, preds), 4),"MAE": round(mean_absolute_error(y_test, preds), 4),
            "R-squared": round(r2_score(y_test, preds), 2),"RMSE": round(root_mean_squared_error(y_test, preds), 3)})
print(results)
df_table = pd.DataFrame(results)

# %%
print(df_table)

# %% [markdown]
# ##  Plot the fitted Line

# %%
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))

# 1. Make predictions for each model one-by-one.
# Notice how Least Squares gets X_test, and the rest get X_test_scaled
pred_ls    = model_ls.predict(X_test)
pred_sgd   = model_sgd.predict(X_test_scaled)
pred_knn   = model_KNN.predict(X_test_scaled)
pred_ridge = model_ridge.predict(X_test_scaled)
pred_lasso = model_lasso.predict(X_test_scaled)

# 2. Draw the black background dots just once
plt.scatter(y_test, pred_ls, color='black', alpha=0.1)

# 3. Use Seaborn to draw the dashed lines for each model.
# scatter=False means "don't draw dots, just the line"
# ci=None means "don't draw shadows around the line"
sns.regplot(x=y_test, y=pred_ls,    scatter=False, label="Least Squares", line_kws={'linestyle': '--'})
sns.regplot(x=y_test, y=pred_sgd,   scatter=False, label="Gradient Descent", line_kws={'linestyle': '--'})
sns.regplot(x=y_test, y=pred_knn,   scatter=False, label="K-Nearest Neighbors", line_kws={'linestyle': '--'})
sns.regplot(x=y_test, y=pred_ridge, scatter=False, label="Ridge", line_kws={'linestyle': '--'})
sns.regplot(x=y_test, y=pred_lasso, scatter=False, label="Lasso", line_kws={'linestyle': '--'})

# 4. Add the legend and show the plot
plt.legend()
plt.show()


