<h1> Banglore-house-prediction- </h1>

    In this notebook, I have made 3 regression models and compared them from each other, 
    and then made a hyperparameter tunning model to get good accuracy.
    
<h3>  1. Packages Used to solve this Regression problem </h3>

       1.import pandas as pd
       2.import numpy as np
       3.from matplotlib import pyplot as plt
       4.%matplotlib inline
       5.import matplotlib 
       6.matplotlib.rcParams["figure.figsize"] = (20,10)
       7.from sklearn.model_selection import train_test_split
       8.from sklearn.linear_model import LinearRegression
       9.from sklearn import metrics
       10.from sklearn.model_selection import ShuffleSplit
       11.from sklearn.model_selection import cross_val_score
       12.from sklearn.tree import DecisionTreeRegressor
       13.from sklearn.ensemble import RandomForestRegressor
       14.from sklearn.model_selection import GridSearchCV
       15.import pickle

<h3>  2. Data Load: Load banglore home prices into a dataframe </h3>

        a. Shape of Dataset --->  (13320, 9)
        b. Drop features that are not required to build our model
        
<h3>  3. Data Cleaning: Handle NA values </h3>

            location       1
            size          16
            total_sqft     0
            bath          73
            price          0
            
<h3>  4. Feature Engineering </h3>

          1. Add new feature(integer) for bhk (Bedrooms Hall Kitchen)
          2. Explore total_sqft feature
          3. Add new feature called price per square feet.
          4. Examine locations which is a categorical variable. We need to apply dimensionality reduction 
             technique here to reduce number of locations.
          
<h3>  5. Dimensionality Reduction </h3>

        1. Any location having less than 10 data points should be tagged as "other" location. 
           This way number of categories can be reduced by huge amount. Later on when we do one hot encoding, 
           it will help us with having fewer dummy columns
      
<h3>  6. Outlier Removal Using Business Logic </h3>

        1. Normally square ft per bedroom is 300 (i.e. 2 bhk apartment is minimum 600 sqft. If you have for example 
           400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier. We will remove 
           such outliers by keeping our minimum thresold per bhk to be 300 sqft.
           
        2. Price per Square feet count plot
           
![Alt Text](https://github.com/Aamir8539/Banglore-house-prediction-/blob/main/download%20(1).png)
      
<h3>  7. Outlier Removal Using Bathrooms Feature </h3>

        2. No.of Bathrooms count plot
           
![Alt Text](https://github.com/Aamir8539/Banglore-house-prediction-/blob/main/download.png)
        
<h3>  8. Use One Hot Encoding For Location </h3>

<h3>  9. Build a Linear Regression Model </h3>

      1. Accuracy -  0.6332077040691115
      2. Mean Absolute Error (MAE): 36.20213019528565
      3. Mean Squared Error (MSE): 5629.063278678316
      4. Root Mean Squared Error (RMSE): 75.0270836343671
      
<h3>  10. Use K Fold cross validation to measure accuracy of our LinearRegression model </h3>

       1. Accuracy - array([ 0.4891095 ,  0.49734726, -0.05754908,  0.30595316,  0.48753056,
                             0.40938694,  0.50612099,  0.47337364,  0.48013987,  0.41902142])
                             
<h3>  11. Build a Decision Tree Modell </h3>

      1. Accuracy - 0.25512119077381146
      2. Mean Absolute Error (MAE): 35.19186836743252
      3. Mean Squared Error (MSE): 11431.455890967827
      4. Root Mean Squared Error (RMSE): 106.91798675137794
      
<h3>  12. Build a Random Forest Regression Model </h3>

      1. Accuracy - 0.4553280573857378
      2. Mean Absolute Error (MAE): 31.23681242536092
      3. Mean Squared Error (MSE): 8358.934648054948
      4. Root Mean Squared Error (RMSE): 91.42720956069341
      
<h3>  13. Hyper parameter tunning with grid serch cv </h3>

<h3>  11. Build a Random Forest Regression model using the hyperparameters </h3>
            
          Accuracy - 0.5972567939184827


<h3>  14. Based on above results we can say that LinearRegression gives the best score. Hence we will use that. </h3>

      

