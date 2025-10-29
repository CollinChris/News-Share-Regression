import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import randint, uniform, zscore
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, accuracy_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#from src.eda import eda as e
 
class eda:
    def __init__(self, name):
        self.name = name
    @staticmethod
    def detect_outliers(df, column, z_thresh=3, plot=True):
        """
        Detects outliers in a numeric column using IQR and Z-score methods.
        
        Parameters:
            df (pd.DataFrame): The input DataFrame.
            column (str): The column to analyze.
            z_thresh (float): Z-score threshold for outlier detection.
            plot (bool): Whether to show a boxplot for visual inspection.
        
        Returns:
            dict: Summary with counts and DataFrames of outliers.
        """
        data = df[column].dropna()
        # IQR method
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        iqr_lower = Q1 - 1.5 * IQR
        iqr_upper = Q3 + 1.5 * IQR
        iqr_outliers = df[(df[column] < iqr_lower) | (df[column] > iqr_upper)]

        # Z-score method
        z_scores = zscore(data)
        z_outliers = df[np.abs(z_scores) > z_thresh]

        # Optional plot
        if plot:
            plt.figure(figsize=(10, 4))
            sns.boxplot(x=df[column])
            plt.title(f'Boxplot of {column}')
            plt.show()

        # Summary
        print(f"\nOutlier Summary for '{column}':")
        print(f"Total rows: {len(df)}")
        print(f"IQR outliers: {len(iqr_outliers)}")
        print(f"Z-score outliers (threshold={z_thresh}): {len(z_outliers)}")
        return {
        'iqr_outliers': iqr_outliers,
        'zscore_outliers': z_outliers,
        'iqr_count': len(iqr_outliers),
        'zscore_count': len(z_outliers)
        }

    @staticmethod
    def bar(df=None, feature=None, hue=None, figsize=(12, 6), palette='pastel'):  
        plt.figure(figsize=figsize)
        ax = sns.countplot(data=df, x=feature, hue=hue, palette=palette)  # Use ax for easier manipulation
        # Rotate x-axis labels for readability
        ax.tick_params(axis='x', rotation=45)
        # Annotate the bars with counts (your original logic)
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10)
        plt.title(f'Count of {feature}')
        plt.tight_layout()  # Adjust layout to prevent cutoff after rotation
        plt.show()

    @staticmethod
    def bar_asc(df=None, feature=None, hue=None, figsize=(12, 6), palette='pastel'):  
        plt.figure(figsize=figsize)
        ax = sns.countplot(data=df, x=feature, hue=hue, palette=palette, order=df[feature].value_counts().index)  
        # Rotate x-axis labels for readability
        ax.tick_params(axis='x', rotation=45)
        # Annotate the bars with counts (your original logic)
        for p in ax.patches:
            ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', fontsize=10)
        plt.title(f'Count of {feature}')
        plt.tight_layout()  # Adjust layout to prevent cutoff after rotation
        plt.show()

    @staticmethod
    def box(df=None, x=None, y=None, figsize=(12, 6), palette='Set2'):
        plt.figure(figsize=figsize)
        sns.boxplot(data=df, x=x, y=y, palette=palette)
        plt.title(f'{y} by {x}')
        plt.xticks(rotation=45)  # Rotate x-axis labels

    @staticmethod
    def hist(df=None, x=None, figsize=(12, 6), kde=True, bins=30, color='skyblue'):
        plt.figure(figsize=figsize)
        sns.histplot(data=df, x=x, kde=kde, bins=bins, color=color)
        plt.title(f'Distribution of {x}')
        plt.xlabel(x)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def scatter(df=None, x=None, y=None, figsize=(12, 6), hue=None):
        plt.figure(figsize=figsize)
        sns.scatterplot(data=df, x=x, y=y, hue=hue)
        plt.title(f'{y} by {x}')  

    @staticmethod
    def spearman(df: pd.DataFrame, numerical_features: list, figsize=(12, 8), cmap='coolwarm'):
        """
        Plots a Spearman correlation heatmap for selected numerical features.

        Parameters:
        - df: pandas DataFrame containing the dataset
        - numerical_features: list of column names to include in the correlation
        - figsize: tuple for figure size
        - cmap: colormap for heatmap
        """
        spearman_vars = df[numerical_features].dropna()
        corr = spearman_vars.corr(method='spearman')

        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap)
        plt.title('Spearman Correlation Matrix')
        plt.tight_layout()
        plt.show()
    @staticmethod

    def pearson(df: pd.DataFrame, numerical_features: list, figsize=(12, 8), cmap='coolwarm'):
        """
        Plots a Pearsons correlation heatmap for selected numerical features.

        Parameters:
        - df: pandas DataFrame containing the dataset
        - numerical_features: list of column names to include in the correlation
        - figsize: tuple for figure size
        - cmap: colormap for heatmap
        """
        pearson_vars = df[numerical_features].dropna()
        corr = pearson_vars.corr(method='pearson')

        plt.figure(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap)
        plt.title('Pearsons Correlation Matrix')
        plt.tight_layout()
        plt.show()
    @staticmethod
    def split_data(df, target):
        # Separate data into features and target variable
        X = df.drop(columns=[target])
        y = df[target]
        return X, y
    
    @staticmethod
    def split_train(X, y):
        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def split_train_class(X, y):
        # Split the data into training, validation, and test sets
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    @staticmethod
    def train_baseline(models, preprocessor, X_train, X_val, y_train, y_val):
        results = []
        for name, model in models.items():
        # Create pipeline with preprocessor and model
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            
            # Fit model on training data
            pipeline.fit(X_train, y_train)
            
            # Predict on validation data
            y_val_pred = pipeline.predict(X_val)
            
            # Calculate metrics on validation set
            mse = mean_squared_error(y_val, y_val_pred)
            mae = mean_absolute_error(y_val, y_val_pred)
            rmse = root_mean_squared_error(y_val, y_val_pred)
            r2 = r2_score(y_val, y_val_pred)
            
            # Residuals for RSE
            residuals = y_val - y_val_pred
            rse = np.std(residuals)  # Approximate RSE as standard deviation of residuals
            
            # R² SD via cross-validation on training data
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
            r2_sd = cv_scores.std() if len(cv_scores) > 1 else 0.0
            
            # Store results
            results.append((name, mse, rmse, mae, rse, r2, r2_sd))
            
            # Print results
            print(f"{name} → MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, RSE: {rse:.2f}, R²: {r2:.2f}, R² SD: {r2_sd:.4f}")

    @staticmethod
    def train_baseline_class(models, preprocessor, X_train, y_train, X_val, y_val):
        results = []
        for name, model in models.items():
            # Create pipeline with preprocessor and model
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            
            # Fit model on training data
            pipeline.fit(X_train, y_train)
            
            # Predict on validation data
            y_val_pred = pipeline.predict(X_val)
            
            # Calculate metrics on validation set
            accuracy = accuracy_score(y_val, y_val_pred)
            precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
            
            # Calculate specificity using confusion matrix
            cm = confusion_matrix(y_val, y_val_pred)
            if len(np.unique(y_val)) == 2:  # Binary classification
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:  # Multi-class classification
                specificities = []
                weights = np.bincount(y_val) / len(y_val)  # Class weights based on support
                for i in range(cm.shape[0]):
                    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])  # TN for class i
                    fp = cm[:, i].sum() - cm[i, i]  # FP for class i
                    specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    specificities.append(specificity_i)
                specificity = np.average(specificities, weights=weights) if specificities else 0.0
            
            # ROC-AUC (only for binary classification)
            roc_auc = None
            if len(np.unique(y_val)) == 2:  # Binary classification
                y_val_pred_proba = pipeline.predict_proba(X_val)[:, 1]
                roc_auc = roc_auc_score(y_val, y_val_pred_proba)
            
            # Cross-validation accuracy on training data
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
            cv_accuracy_std = cv_scores.std() if len(cv_scores) > 1 else 0.0
            
            # Calculate and display confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            fig, ax = plt.subplots()
            disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
            plt.title(f"Confusion Matrix - {name}")
            plt.grid(False)
            plt.show()
            
            # Store results
            results.append((name, accuracy, precision, recall, f1, specificity, roc_auc, cv_accuracy_std))
            
            # Print results
            roc_auc_str = f"{roc_auc:.2f}" if roc_auc is not None else "N/A"
            print(f"{name} → Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, "
                f"Recall: {recall:.2f}, F1: {f1:.2f}, Specificity: {specificity:.2f}, "
                f"ROC-AUC: {roc_auc_str}, CV Accuracy SD: {cv_accuracy_std:.4f}")
        
        return results

    @staticmethod
    def tune_models_random(models_params, X_train, y_train, X_val, y_val, preprocessor, n_iter=20): # Function to perform random search and evaluate
        results = {}
        
        for name, mp in models_params.items():
            # Create pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', mp['model'])])
            
            # Randomized search with cross-validation
            random_search = RandomizedSearchCV(
                pipeline,
                mp['params'],
                n_iter=n_iter,  # Number of random combinations to try
                cv=5,
                scoring='r2',
                n_jobs=-1,  # Use all available cores
                random_state=42,  # For reproducibility
                verbose=1
            )
            
            # Fit random search
            random_search.fit(X_train, y_train)
            
            # Best model and parameters
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            
            # Predict on validation set with best model
            y_val_pred = best_model.predict(X_val)
            
            # Calculate additional metrics on validation set
            mse = mean_squared_error(y_val, y_val_pred)
            mae = mean_absolute_error(y_val, y_val_pred)
            rmse = np.sqrt(mse)
            residuals = y_val - y_val_pred
            rse = np.std(residuals)
            r2 = r2_score(y_val, y_val_pred)
            
            # Store results
            results[name] = {
                'best_params': best_params,
                'best_cv_score': best_score,
                'val_mse': mse,
                'val_mae': mae,
                'val_rmse': rmse,
                'val_rse': rse,
                'val_r2': r2
            }
            
            # Print results
            print(f"\n{name} Tuning Results:")
            print(f"Best Parameters: {best_params}")
            print(f"Best CV R²: {best_score:.4f}")
            print(f"Validation MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, RSE: {rse:.2f}, R²: {r2:.2f}")        
        return results
    
    @staticmethod
    def tune_models_grid(models_params, X_train, y_train, X_val, y_val, preprocessor, cv=5):
        """
        Perform GridSearchCV for hyperparameter tuning and evaluate models on validation set.
        
        Parameters:
        - models_params: Dict with model names, model instances, and parameter grids.
        - X_train, y_train: Training features and target (log-transformed).
        - X_val, y_val: Validation features and target (log-transformed).
        - preprocessor: ColumnTransformer for preprocessing.
        - cv: Number of cross-validation folds (default: 5).
        
        Returns:
        - results: Dict with best model, parameters, CV score, and validation metrics.
        """
        results = {}
        
        for name, mp in models_params.items():
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', mp['model'])])
            
            grid_search = GridSearchCV(
                pipeline,
                mp['params'],
                cv=cv,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            y_val_pred = best_model.predict(X_val)
            
            mse = mean_squared_error(y_val, y_val_pred)
            mae = mean_absolute_error(y_val, y_val_pred)
            rmse = np.sqrt(mse)
            residuals = y_val - y_val_pred
            rse = np.std(residuals)
            r2 = r2_score(y_val, y_val_pred)
            
            results[name] = {
                'best_model': best_model,
                'best_params': best_params,
                'best_cv_score': best_score,
                'val_mse': mse,
                'val_mae': mae,
                'val_rmse': rmse,
                'val_rse': rse,
                'val_r2': r2,
            }
            
            print(f"\n{name} Tuning Results (GridSearchCV):")
            print(f"Best Parameters: {best_params}")
            print(f"Best CV R²: {best_score:.4f}")
            print(f"Validation MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, RSE: {rse:.2f}, R²: {r2:.2f}")
            print(f"Validation MAE (original scale): {mae_orig:.2f}")
        
        return results
    
    @staticmethod
    def tune_mclass_grid(models_params, X_train, y_train, X_val, y_val, preprocessor):
        results = {}
        
        for name, mp in models_params.items():
            # Create pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', mp['model'])])
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                pipeline,
                mp['param_grid'],
                cv=5,
                scoring='accuracy',  # Use accuracy for classification
                n_jobs=-1,  # Use all available cores
                verbose=1
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            # Best model and parameters
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            # Predict on validation set with best model
            y_val_pred = best_model.predict(X_val)
            
            # Calculate metrics on validation set
            accuracy = accuracy_score(y_val, y_val_pred)
            precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
            
            # Calculate specificity using confusion matrix
            cm = confusion_matrix(y_val, y_val_pred)
            if len(np.unique(y_val)) == 2:  # Binary classification
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:  # Multi-class classification
                specificities = []
                weights = np.bincount(y_val) / len(y_val)  # Class weights based on support
                for i in range(cm.shape[0]):
                    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])  # TN for class i
                    fp = cm[:, i].sum() - cm[i, i]  # FP for class i
                    specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    specificities.append(specificity_i)
                specificity = np.average(specificities, weights=weights) if specificities else 0.0
            
            # ROC-AUC (only for binary classification)
            roc_auc = None
            if len(np.unique(y_val)) == 2:  # Binary classification
                y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
                roc_auc = roc_auc_score(y_val, y_val_pred_proba)
            
            # Store results
            results[name] = {
                'best_params': best_params,
                'best_cv_score': best_score,
                'val_accuracy': accuracy,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1,
                'val_specificity': specificity,
                'val_roc_auc': roc_auc
            }
            
            # Print results
            roc_auc_str = f"{roc_auc:.2f}" if roc_auc is not None else "N/A"
            print(f"\n{name} Tuning Results:")
            print(f"Best Parameters: {best_params}")
            print(f"Best CV Accuracy: {best_score:.4f}")
            print(f"Validation Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, "
                f"Recall: {recall:.2f}, F1: {f1:.2f}, Specificity: {specificity:.2f}, "
                f"ROC-AUC: {roc_auc_str}")
        
        return results

    @staticmethod
    def tune_mclass_random(models_params, X_train, y_train, X_val, y_val, preprocessor, n_iter=20):
        results = {}
        
        for name, mp in models_params.items():
            # Create pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', mp['model'])])
            
            # Randomized search with cross-validation
            random_search = RandomizedSearchCV(
                pipeline,
                mp['param_distributions'],
                n_iter=n_iter,  # Number of random combinations to try
                cv=5,
                scoring='accuracy',  # Use accuracy for classification
                n_jobs=-1,  # Use all available cores
                random_state=42,  # For reproducibility
                verbose=1
            )
            
            # Fit random search
            random_search.fit(X_train, y_train)
            
            # Best model and parameters
            best_model = random_search.best_estimator_
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            
            # Predict on validation set with best model
            y_val_pred = best_model.predict(X_val)
            
            # Calculate metrics on validation set
            accuracy = accuracy_score(y_val, y_val_pred)
            precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)
            
            # Calculate specificity using confusion matrix
            cm = confusion_matrix(y_val, y_val_pred)
            if len(np.unique(y_val)) == 2:  # Binary classification
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:  # Multi-class classification
                specificities = []
                weights = np.bincount(y_val) / len(y_val)  # Class weights based on support
                for i in range(cm.shape[0]):
                    tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])  # TN for class i
                    fp = cm[:, i].sum() - cm[i, i]  # FP for class i
                    specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                    specificities.append(specificity_i)
                specificity = np.average(specificities, weights=weights) if specificities else 0.0
            
            # ROC-AUC (only for binary classification)
            roc_auc = None
            if len(np.unique(y_val)) == 2:  # Binary classification
                y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
                roc_auc = roc_auc_score(y_val, y_val_pred_proba)
            
            # Store results
            results[name] = {
                'best_params': best_params,
                'best_cv_score': best_score,
                'val_accuracy': accuracy,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1,
                'val_specificity': specificity,
                'val_roc_auc': roc_auc
            }
            
            # Print results
            roc_auc_str = f"{roc_auc:.2f}" if roc_auc is not None else "N/A"
            print(f"\n{name} Tuning Results:")
            print(f"Best Parameters: {best_params}")
            print(f"Best CV Accuracy: {best_score:.4f}")
            print(f"Validation Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, "
                f"Recall: {recall:.2f}, F1: {f1:.2f}, Specificity: {specificity:.2f}, "
                f"ROC-AUC: {roc_auc_str}")
        
        return results

    @staticmethod
    def evaluate_best_mclass(results, models_params, X_train, y_train, X_val, y_val, X_test, y_test, preprocessor):
        """
        Evaluate the best model from GridSearchCV results on the test set.
        
        Parameters:
        - results: Dictionary containing GridSearchCV results (model names, best_params, best_cv_score, etc.)
        - models_params: Dictionary containing model instances and their parameter grids
        - X_train, y_train: Training data and labels
        - X_val, y_val: Validation data and labels
        - X_test, y_test: Test data and labels
        - preprocessor: Preprocessor for the pipeline
        
        Returns:
        - Dictionary with test set metrics and best model information
        """
        # Combine training and validation sets for final training
        if isinstance(X_train, np.ndarray):
            X_train_val = np.vstack([X_train, X_val])
            y_train_val = np.hstack([y_train, y_val])
        elif isinstance(X_train, pd.DataFrame):
            X_train_val = pd.concat([X_train, X_val], axis=0, ignore_index=True)
            y_train_val = pd.concat([y_train, y_val], axis=0, ignore_index=True) if isinstance(y_train, pd.Series) else np.hstack([y_train, y_val])
        else:
            raise ValueError("X_train must be a NumPy array or Pandas DataFrame")
        
        # Find the best model based on best_cv_score
        best_model_name = max(results, key=lambda x: results[x]['best_cv_score'])
        best_model_info = results[best_model_name]
        best_params = best_model_info['best_params']
        best_cv_score = best_model_info['best_cv_score']
        
        # Get the base model instance from models_params
        base_model = models_params[best_model_name]['model']
        
        # Create pipeline with preprocessor and best model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', base_model)])
        
        # Set the best parameters
        pipeline.set_params(**best_params)
        
        # Fit the model on combined train+val data
        pipeline.fit(X_train_val, y_train_val)
        
        # Predict on test set
        y_test_pred = pipeline.predict(X_test)
        
        # Calculate metrics on test set
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        
        # Calculate specificity using confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        if len(np.unique(y_test)) == 2:  # Binary classification
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:  # Multi-class classification
            specificities = []
            weights = np.bincount(y_test) / len(y_test)  # Class weights based on support
            for i in range(cm.shape[0]):
                tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])  # TN for class i
                fp = cm[:, i].sum() - cm[i, i]  # FP for class i
                specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                specificities.append(specificity_i)
            specificity = np.average(specificities, weights=weights) if specificities else 0.0
        
        # ROC-AUC (only for binary classification)
        roc_auc = None
        if len(np.unique(y_test)) == 2:  # Binary classification
            y_test_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        
        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
        plt.title(f"Confusion Matrix - {best_model_name} (Test Set)")
        plt.grid(False)
        plt.show()
        
        # Store results
        test_results = {
            'best_model': best_model_name,
            'best_params': best_params,
            'best_cv_score': best_cv_score,
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'test_specificity': specificity,
            'test_roc_auc': roc_auc
        }
        
        # Print results
        roc_auc_str = f"{roc_auc:.2f}" if roc_auc is not None else "N/A"
        print(f"\nBest Model: {best_model_name}")
        print(f"Best Parameters: {best_params}")
        print(f"Best CV Accuracy (from GridSearch): {best_cv_score:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, "
            f"Recall: {recall:.2f}, F1: {f1:.2f}, Specificity: {specificity:.2f}, "
            f"ROC-AUC: {roc_auc_str}")
        
        return test_results

    @staticmethod
    def evaluate_best_models(results, X_train, y_train, X_test, y_test, preprocessor):
        model_configs = {}
        
        # Extract and configure models from results
        for name, res in results.items():
            params = res['best_params']
            if name == 'RandomForest':
                model_configs[name] = {
                    'model': RandomForestRegressor(
                        max_depth=params.get('model__max_depth'),
                        min_samples_leaf=params.get('model__min_samples_leaf'),
                        min_samples_split=params.get('model__min_samples_split'),
                        n_estimators=params.get('model__n_estimators'),
                        random_state=42
                    ),
                    'name': name
                }
            elif name == 'GradientBoosting':
                model_configs[name] = {
                    'model': GradientBoostingRegressor(
                        learning_rate=params.get('model__learning_rate'),
                        max_depth=params.get('model__max_depth'),
                        min_samples_split=params.get('model__min_samples_split'),
                        n_estimators=params.get('model__n_estimators'),
                        random_state=42
                    ),
                    'name': name
                }
            elif name == 'SVR':
                model_configs[name] = {
                    'model': SVR(
                        kernel=params.get('model__kernel', 'rbf'),
                        C=params.get('model__C', 1.0),
                        epsilon=params.get('model__epsilon', 0.1),
                        gamma=params.get('model__gamma', 'scale')
                    ),
                    'name': name
                }

        results_test = {}
        
        for model_info in model_configs.values():
            name = model_info['name']
            model = model_info['model']
            
            # Create pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            
            # Fit on training data
            pipeline.fit(X_train, y_train)
            
            # Predict on test data
            y_test_pred = pipeline.predict(X_test)
            
            # Calculate metrics on test set
            mse = mean_squared_error(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            rmse = np.sqrt(mse)
            residuals = y_test - y_test_pred
            rse = np.std(residuals)
            r2 = r2_score(y_test, y_test_pred)
            
            # Store results
            results_test[name] = {
                'test_mse': mse,
                'test_mae': mae,
                'test_rmse': rmse,
                'test_rse': rse,
                'test_r2': r2
            }
            
            # Print results
            print(f"\n{name} Test Results:")
            print(f"Test MSE: {mse:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, RSE: {rse:.2f}, R²: {r2:.2f}")       
        return results_test