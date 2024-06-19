import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def read_data_from_file(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # Extract header
        headers = lines[0].strip().split('\t')[2:]
        for line in lines[1:]:
            items = line.strip().split('\t')
            gene_id = items[0]
            expression_values = [float(value) for value in items[2:]]
            data_dict[gene_id] = dict(zip(headers, expression_values))
            # print(gene_id)
    # print(data_dict["224685_at"])
    return data_dict


# Helper function to prepare data and apply PCA
def prepare_data_and_apply_pca(data, threshold=1000, n_components=10, test_size=0.2, random_state=42):
    gene_ids = list(data.keys())
    headers = list(data[gene_ids[0]].keys())
    # print(headers)
    temp1 = headers[0]
    # for gene_id in gene_ids:
    #     values_length = len(data[gene_id].values())
    #     print(f"Gene ID: {gene_id}, Values Length: {values_length}")

    # Extract expression values
    temp=[list(data[gene_id].values()) for gene_id in gene_ids]
    
    temp=temp[:-1]
    expression_values = np.array(temp)
    mean = np.mean(expression_values)
    threshold = mean
    
    # Generate binary labels based on a threshold
    labels = np.array([1 if val > threshold else 0 for val in expression_values[:, headers.index(temp1)]])

    # Standardize the data
    scaler = StandardScaler()
    scaled_expression_values = scaler.fit_transform(expression_values)

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_features = pca.fit_transform(scaled_expression_values)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(pca_features, labels, test_size=test_size, random_state=random_state)

    # Train SVM model
    svm_model = SVC(kernel='linear', random_state=random_state)
    svm_model.fit(X_train, y_train)

    # Make predictions
    y_pred = svm_model.predict(X_test)

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Example usage
file_path = r"give path"
data = read_data_from_file(file_path)

# Apply PCA and SVM
prepare_data_and_apply_pca(data, threshold=100, n_components=5)
