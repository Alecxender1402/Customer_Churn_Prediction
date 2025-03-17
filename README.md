# Customer Churn Predication

A deep learning application that predicts customer churn probability using neural networks. Built with TensorFlow and Streamlit for interactive predictions.

## Features ✨
- **Real-time Predictions**: Instant churn probability calculation
- **Interactive Interface**: User-friendly web form inputs
- **trained Model**: TensorFlow neural network with 3 dense layers
- **Data Preprocessing**: Built-in feature scaling and encoding
- **Probability Display**: Clear churn likelihood percentage

## Project Structure 📂
.
├── app.py                         # Main application interface
├── churn_model.h5                 # Trained neural network model
├── index.ipynb                    # Model training notebook
├── label_encoder_gender.pkl       # Gender encoding
├── one_hot_encoder_geography.pkl  # Geography encoding
├── scalar.pkl                     # Feature scaler
└── requirement.txt                # Dependency list

## Installation 🛠️

### Prerequisites :-
- Python 3.7+
- pip package manager

### Steps :-
1. Clone the repository
```bash
git clone https://github.com/Alecxender1402/Customer_Churn_Prediction.git
```
2. Install dependencies:
```bash
pip install -r requirement.txt 
```

### Configuration ⚙️
Ensure these files are present in root directory:
- **churn_model.h5**: trained TensorFlow model
- **scalar.pkl**: Feature scaler
- **label_encoder_gender.pkl**: Gender encoder
- **one_hot_encoder_geography.pkl**: Geography encoder

### Usage 🚀
1. Start the app:-
```bash
streamlit run app.py
```
2. Fill in customer details:
   - Select geography from dropdown 🌍
   - Choose gender 👫
   - Adjust numerical sliders 🔢
   - Set product/service parameters 📅
3. View results:
   - Immediate churn classification 🚨
   - Probability percentage 📈
   - Clear visual feedback ✅

### Dependencies 📦

- **Core Framework**: Streamlit, TensorFlow
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Support**: ipykernel, tensorboard


