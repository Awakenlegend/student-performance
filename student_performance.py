
# In[1]:
import numpy as np #used for math operation
import pandas as pd # used data analysis

# In[2]:
df=pd.read_csv('/content/StudentsPerformance.csv')#import dataset

# In[3]:
df

# In[4]:
df.head()
df.info()

# In[5]:
df.describe()

# In[6]:
df["TotalScore"] = df["math score"] + df["reading score"] + df["writing score"] # feature engineering to calculate the total score

# In[7]:
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()#categorical encoder


# In[8]:
df["gender"] = encoder.fit_transform(df["gender"])# male as 1 and female as 0
df["lunch"] = encoder.fit_transform(df["lunch"]) #free 0 and standard 1
df["test preparation course"] = encoder.fit_transform(df["test preparation course"])# none as 0 and complted as 1
df["parental level of education"] = encoder.fit_transform(df["parental level of education"])# bachelor's degree as 0,master's degree as 1,high school as 2,associate's degree as 3
df

# In[9]:
# Features and Target
X = df[[
    "gender",
    "lunch",
    "test preparation course",
    "parental level of education",
    "reading score",
    "writing score"
]]
y = df["math score"]

# In[10]:
from sklearn.model_selection import train_test_split

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)# here 20% is test and 80% is train

# In[11]:
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Models
models = {
    "Linear Regression": LinearRegression(),#fits a straight line to predict values
    "Random Forest": RandomForestRegressor(),#uses many decision trees and averages their predictions
    "Gradient Boosting": GradientBoostingRegressor()#builds trees sequentially to correct previous errors.
}

# In[12]:
# Training and Evaluation
for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Calculate MSE and then take its square root to get RMSE
    mse = mean_squared_error(y_test, predictions)#it tells how far our value from actual value
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)#it tells how good our model is

    print(f"\n{name}")
    print("RMSE:", rmse)
    print("R2 Score:", r2)