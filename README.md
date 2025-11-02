# Heart Disease Prediction - Machine Learning Project

A comprehensive machine learning project for predicting 10-year risk of Coronary Heart Disease (CHD) using the Framingham Heart Study dataset.

## 📊 Dataset

- **Dataset**: Framingham Heart Disease Study
- **Samples**: 4,240 patients
- **Features**: 15 features (demographics, lifestyle, clinical measurements)
- **Target**: Binary classification (CHD risk: Yes/No)
- **Class Distribution**: 84.8% No CHD, 15.2% CHD Risk (imbalanced)

## 🔬 Project Structure

### Part 1: Data Understanding & Problem Definition
- Data loading and exploration
- Missing values analysis
- Target variable identification
- Feature identification

### Part 2: Data Cleaning & Pre-processing
- Missing value imputation (Median)
- Outlier detection using IQR method
- Feature scaling (StandardScaler)
- Train-test split (70-30, stratified)

### Part 3: Exploratory Data Analysis (EDA)
- Correlation analysis and heatmap
- Feature distributions by target class
- Identifying predictive features

### Part 4: Supervised Learning
- **Models Trained**: 7 algorithms
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
  - Gradient Boosting
  - Decision Tree
  - Naive Bayes

- **Best Model**: SVM (84.98% accuracy)

### Part 5: Unsupervised Learning
- **K-Means Clustering**: Patient risk profile grouping
- **Anomaly Detection**: Isolation Forest for unusual patients
- **Feature Enhancement**: Using clusters as additional features

## 📈 Results

- **Top Model Performance**: SVM with 84.98% accuracy
- **Models Comparison**: All 7 models evaluated with metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- **Cluster Analysis**: Identified distinct risk profiles
- **Anomaly Detection**: Detected unusual patient patterns

## 🛠️ Technologies Used

- **Python 3.13**
- **Libraries**:
  - pandas, numpy
  - matplotlib, seaborn
  - scikit-learn
  - scipy

## 📝 Key Insights

1. Age, blood pressure, and cholesterol are top predictors
2. Class imbalance handled with stratified sampling
3. Unsupervised learning revealed natural patient groupings
4. Cluster information can enhance supervised models

## 🚀 Usage

1. Clone the repository
2. Open `heart_disease_analysis.ipynb` in Jupyter Notebook or VS Code
3. Ensure all required libraries are installed
4. Run cells sequentially

## 📦 Files

- `heart_disease_analysis.ipynb` - Complete ML pipeline notebook
- `framingham.csv` - Dataset

## 👤 Author

SMariamD

## 📄 License

This project is open source and available for educational purposes.

