# Fright-Rate-Prediction-Challenge
 ## Data Splitting Methodology
 
To account for the specialized temporal and spatial nature of the datasets, the data splitting methodology avoids complete random splitting—which could disrupt critical data indications—and instead employs a temporal split that accounts for seasonality alongside spatial grouping based on latitude and longitude coordinates. Shuffling is strictly restricted within these spatial clusters to preserve geographic integrity, while monthly groupings are introduced to evaluate seasonality effects across different equipment types (Dry Van, Flatbed, and Reefer), each requiring tailored seasonal treatments due to their distinct cargo characteristics and operational timeframes.

 ## Data Preprocessing

Prior to feature engineering, comprehensive data validation was conducted to enforce structural integrity by verifying the uniqueness of ID columns, standardizing date formats, and cross-checking geographic coordinates for spatial consistency. For feature engineering, spatial aggregation was performed using latitude and longitude coordinates alongside encoding mechanisms applied to string and categorical variables. Although all features were initially retained to establish a complete baseline, a subsequent feature selection process was executed to optimize the feature set for final modeling.

## Model Selection

For the model selection strategy, three ensemble learning algorithms from Scikit-learn were evaluated: HistGradientBoosting, a fast, memory-efficient histogram-based gradient boosting method that discretizes continuous features into 256 integer bins to mitigate overfitting on large datasets; RandomForest, a bagging architecture that trains deep, independent decision trees in parallel using feature randomization and aggregates their predictions via majority voting or averaging; and standard GradientBoosting, a sequential boosting technique that minimizes residual errors through tree depth constraints, subsampling, and shrinkage. Although gradient boosting frameworks such as LightGBM, CatBoost, and XGBoost were initially considered, opting exclusively for Scikit-learn's ensemble estimators streamlined the data preprocessing pipeline by allowing a unified encoding scheme across all models rather than configuring framework-specific categorical handlers, thereby accelerating implementation and iteration efficiency.

## Results

Among the evaluated algorithms, HistGradientBoosting with 300 trees achieved the best overall performance, delivering the highest predictive accuracy (R^2 = 0.8293$) alongside the lowest error rates (RMSE = 609.57$, MAE = 202.35$). Increasing the number of trees in HistGradientBoosting demonstrated a clear progression from severe underfitting at 5 trees (R^2 = 0.0758$) to competitive results at 100 trees (R^2 = 0.7081$) and optimal convergence at 300 trees. RandomForest closely followed as the second-best model (R^2 = 0.8187, RMSE = 628.06), while standard GradientBoosting performed similarly to the 100-tree HistGradientBoosting variant (R^2 = 0.7277$). Overall, HistGradientBoosting (300 trees) proved to be the most effective and efficient solution for this dataset.


