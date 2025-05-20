Intrusion Detection System (IDS) Using Machine Learning
Overview
This project implements an Intrusion Detection System (IDS) using machine learning to classify network traffic as either normal or an attack. The system is built using the KDD Cup 1999 dataset and employs an ensemble of LightGBM and Random Forest models, with feature selection optimized using a Correlation-based Feature Selection (CFS) method and a simplified Bat Algorithm.
Directory Structure
Intrusion-Detection-System/
├── KDDTrain+.txt         # Training dataset (download from link below)
├── KDDTest+.txt          # Test dataset (download from link below)
├── intrusion_detection.py # Main script
├── ids_results.txt       # Performance metrics
├── confusion_matrix.png  # Confusion matrix plot
├── .gitignore
├── README.md
└── requirements.txt

Note: The KDD Cup 1999 dataset files (KDDTrain+.txt and KDDTest+.txt) are not included in this repository due to GitHub’s file size limits. You can download them from the UCI Machine Learning Repository. Place the files in the project root directory before running the script.
Prerequisites

Python 3.6 or higher
Git (for cloning the repository)
Required Python libraries (listed in requirements.txt)

Installation

Clone the Repository:
git clone https://github.com/<your-username>/Intrusion-Detection-System.git
cd Intrusion-Detection-System


Download the Dataset:

Download KDDTrain+.txt and KDDTest+.txt from here.
Place the files in the project root directory (same directory as intrusion_detection.py).


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:
pip install -r requirements.txt



Usage

Run the Script:
python intrusion_detection.py


Output:

The script will print performance metrics (accuracy, precision, recall, F1-score) and a confusion matrix to the console.
Results are saved to ids_results.txt.
A confusion matrix plot is saved as confusion_matrix.png.



Results
The model achieved the following performance on the test set (update with your actual metrics from ids_results.txt):

Intrusion Detection System (IDS) Results
======================================
Accuracy:  0.9297
Precision: 0.9986
Recall:    0.9140
F1-Score:  0.9544
AUC Score: 0.9777

Confusion Matrix:
[[True Normal  False Attack]
 [False Normal True Attack]]
[[ 60271    322]
 [ 21546 228890]]

Methodology

Data Preprocessing:
Encoded categorical features (protocol_type, service, flag) using LabelEncoder.
Handled missing values by filling with medians.
Scaled features using StandardScaler.


Feature Selection:
Used CFS with SelectKBest to select the top 20 features.
Optimized feature selection with a simplified Bat Algorithm.


Model:
Ensemble of LightGBM and Random Forest.
Combined predictions using majority voting.



License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

The KDD Cup 1999 dataset was used for training and testing.


