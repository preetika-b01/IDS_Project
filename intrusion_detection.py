import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# Load and preprocess the dataset
def load_data(train_path='KDDTrain+.txt', test_path='KDDTest+.txt'):
    columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
               'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
               'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
               'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
               'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
               'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
               'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
               'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
               'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label'])
    dtype_dict = {'label': str}
    train_data = pd.read_csv(train_path, names=columns, dtype=dtype_dict)
    test_data = pd.read_csv(test_path, names=columns, dtype=dtype_dict)
    return train_data, test_data

# Preprocess the data: encode categorical features, handle missing values, and scale numerical features
def preprocess_data(train_data, test_data):
    combined_data = pd.concat([train_data, test_data], axis=0)
    combined_data['label'] = combined_data['label'].astype(str).str.rstrip('.')

    categorical_cols = ['protocol_type', 'service', 'flag']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        combined_data[col] = le.fit_transform(combined_data[col].astype(str))
        label_encoders[col] = le

    train_data = combined_data.iloc[:len(train_data)].copy()
    test_data = combined_data.iloc[len(train_data):].copy()

    train_data['label'] = train_data['label'].apply(lambda x: 0 if x == 'normal' else 1)
    test_data['label'] = test_data['label'].apply(lambda x: 0 if x == 'normal' else 1)

    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_test = test_data.drop('label', axis=1)
    y_test = test_data['label']

    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    print("NaN values in X_train before filling:", X_train.isna().any().any())
    print("NaN values in X_test before filling:", X_test.isna().any().any())

    X_train = X_train.fillna(X_train.median(numeric_only=True)).fillna(0)
    X_test = X_test.fillna(X_test.median(numeric_only=True)).fillna(0)

    print("NaN values in X_train after filling:", X_train.isna().any().any())
    print("NaN values in X_test after filling:", X_test.isna().any().any())

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

# Simplified CFS (Correlation-based Feature Selection) using SelectKBest
def cfs_feature_selection(X, y, k=20):
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)
    return X_selected, selected_indices, selector

# Simplified Bat Algorithm for optimizing feature selection
def bat_algorithm(X, y, selected_indices, n_features, n_bats=10, max_iter=10):
    population = np.zeros((n_bats, n_features))
    for i in range(n_bats):
        population[i, selected_indices] = 1
        flip_indices = np.random.choice(n_features, size=5, replace=False)
        population[i, flip_indices] = 1 - population[i, flip_indices]
    best_fitness = -np.inf
    best_solution = population[0]
    for _ in range(max_iter):
        for i in range(n_bats):
            mask = population[i].astype(bool)
            X_subset = X[:, mask]
            X_train_sub, X_val_sub, y_train, y_val = train_test_split(X_subset, y, test_size=0.2, random_state=42)
            clf = RandomForestClassifier(n_estimators=10, random_state=42)
            clf.fit(X_train_sub, y_train)
            fitness = clf.score(X_val_sub, y_val)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = population[i].copy()
            flip_indices = np.random.choice(n_features, size=3, replace=False)
            population[i, flip_indices] = 1 - population[i, flip_indices]
    return best_solution.astype(bool)

# Train ensemble model with LightGBM and Random Forest
def train_ensemble_model(X_train, X_test, y_train, y_test, selected_features):
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    lgb_train = lgb.Dataset(X_train_selected, label=y_train)
    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9
    }
    lgb_model = lgb.train(params, lgb_train, num_boost_round=100)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_selected, y_train)
    lgb_pred = lgb_model.predict(X_test_selected)
    lgb_pred = (lgb_pred > 0.5).astype(int)
    rf_pred = rf_model.predict(X_test_selected)
    ensemble_pred = np.array([1 if (lgb_pred[i] + rf_pred[i]) >= 1 else 0 for i in range(len(lgb_pred))])
    accuracy = accuracy_score(y_test, ensemble_pred)
    precision = precision_score(y_test, ensemble_pred)
    recall = recall_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred)
    conf_matrix = confusion_matrix(y_test, ensemble_pred)
    return accuracy, precision, recall, f1, conf_matrix, lgb_model, rf_model

# Plot confusion matrix and save it
def plot_confusion_matrix(conf_matrix, output_dir='C:/Users/PREETIKA BHARDWAJ/IDS_Project'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.title('Confusion Matrix - Intrusion Detection System')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()

# Main execution
def main():
    print("======================================")
    print("Intrusion Detection System (IDS) Demo")
    print("======================================")
    print("Loading and preprocessing data...\n")

    train_data, test_data = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(train_data, test_data)

    print("\nPerforming feature selection using CFS and Bat Algorithm...\n")
    X_train_selected, selected_indices, selector = cfs_feature_selection(X_train, y_train, k=20)
    n_features = X_train.shape[1]
    selected_features = bat_algorithm(X_train, y_train, selected_indices, n_features)

    print("Training ensemble model (LightGBM + Random Forest)...\n")
    accuracy, precision, recall, f1, conf_matrix, lgb_model, rf_model = train_ensemble_model(X_train, X_test, y_train, y_test, selected_features)

    # Format and display results
    print("======================================")
    print("Model Performance Results")
    print("======================================")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print("[[True Normal  False Attack]")
    print(f" [False Normal True Attack]]")
    print(conf_matrix)

    # Save results to a file
    with open('C:/Users/PREETIKA BHARDWAJ/IDS_Project/ids_results.txt', 'w') as f:
        f.write("Intrusion Detection System (IDS) Results\n")
        f.write("======================================\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write("[[True Normal  False Attack]\n")
        f.write(" [False Normal True Attack]]\n")
        f.write(str(conf_matrix))

    # Plot and save confusion matrix
    print("\nGenerating confusion matrix plot...")
    plot_confusion_matrix(conf_matrix)
    print("Confusion matrix plot saved as 'confusion_matrix.png' in the project directory.")

if __name__ == "__main__":
    main()