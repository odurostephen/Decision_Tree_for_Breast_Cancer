import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import warnings

# Suppressing warnings for clarity and ignoring FitFailedWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Load the dataset
df = pd.read_csv('data.csv')
df.rename({"Unnamed: 32": "a"}, axis="columns", inplace=True)
df.drop(["a"], axis=1, inplace=True)

y = df['diagnosis'].values  # Target variable
X = df.drop('diagnosis', axis=1).values  # Feature variables

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

# Normalization
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Model building
model = DecisionTreeClassifier()

# Tuning Parameters with GridSearchCV and k-fold cross-validation
parameters = {'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
              'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10],
              'max_features': ['auto', 'sqrt', 'log2']}

k_fold = KFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(model, parameters, cv=k_fold, scoring='accuracy')

# Use try-except block to catch warnings and prevent printing
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=Warning)
        grid_search.fit(X_train_sc, y_train)  # Model Fitting
except warnings.FitFailedWarning as e:
    print("Error occurred during model fitting:", e)

# Extracting best hyperparameters
best_min_samples_split = grid_search.best_params_['min_samples_split']
best_min_samples_leaf = grid_search.best_params_['min_samples_leaf']
best_max_features = grid_search.best_params_['max_features']

# After hyperparameter tuning
best_model = grid_search.best_estimator_
best_model.fit(X_train_sc, y_train)
y_pred_dt = best_model.predict(X_test_sc)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# Print results
print("Best Hyperparameters:")
print(f"min_samples_split: {best_min_samples_split}")
print(f"min_samples_leaf: {best_min_samples_leaf}")
print(f"max_features: {best_max_features}")

# Classification Report (includes sensitivity and specificity)
print(classification_report(y_test, y_pred_dt))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_dt)

# Calculate Sensitivity and Specificity
sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
specificity = conf_matrix[0, 0] / (conf_matrix[0, 1] + conf_matrix[0, 0])

# Print additional metrics
print("Accuracy on Test Data:", accuracy_dt)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Confusion Matrix:")
print(pd.DataFrame(conf_matrix, columns=['Benign', 'Malignant'], index=['Benign', 'Malignant']))
