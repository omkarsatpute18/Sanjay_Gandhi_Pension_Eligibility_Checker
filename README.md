# 🏛️ AI-Based Sanjay Gandhi Niradhar Pension Eligibility Prediction System

> An AI-powered web application that predicts whether an applicant is eligible for the **Sanjay Gandhi Niradhar Pension Yojana** using Machine Learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-purple.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

# 📌 Project Overview

The **AI-Based Sanjay Gandhi Niradhar Pension Eligibility Prediction System** is a Machine Learning-based web application that helps predict whether an applicant is eligible for the Sanjay Gandhi Niradhar Pension Scheme.

The system analyzes multiple applicant details such as age, annual family income, BPL status, disability percentage, bank account availability, certificates, and other government eligibility criteria to generate a prediction.

The project is developed for educational purposes to demonstrate the practical implementation of Artificial Intelligence and Machine Learning in Government Welfare Schemes.

---

# ✨ Features

- 🤖 AI-based eligibility prediction
- 🏛 Government portal-inspired responsive UI
- 📊 Confidence score for every prediction
- 📋 Detailed reason for prediction
- 📁 CSV input and output support
- ⚡ Fast prediction using Random Forest
- 📱 Mobile responsive design
- 🔒 Clean and professional interface
- 🎯 High prediction accuracy
- 💻 Easy Flask integration

---

# 🧠 Machine Learning Model

Model Used:

- Random Forest Classifier

Libraries:

- Scikit-Learn
- Pandas
- Joblib
- NumPy

Model Parameters:

```python
RandomForestClassifier(
    n_estimators=30,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
```

---

# 📂 Project Structure

```
AI-Pension-Prediction/
│
├── app.py
├── main.py
├── index.html
├── model.pkl
├── encoder.pkl
├── feature_columns.pkl
├── cat_columns.pkl
├── sanjay_gandhi_pension_eligibility.csv
├── input.csv
├── output.csv
├── requirements.txt
├── README.md
└── assets/
```

---

# 📊 Dataset

The dataset contains applicant information such as:

- Applicant ID
- Applicant Name
- Gender
- Age
- Marital Status
- Category
- Annual Family Income
- BPL Status
- Disability Percentage
- Widow Status
- Orphan Status
- Chronic Illness
- Bank Account
- Aadhaar Linked
- Income Certificate
- Residence Certificate
- District
- Rural / Urban
- Occupation
- Family Members
- Land Holding
- House Type
- Education
- Mobile Number
- Eligible (Target)

---

# ⚙️ Technologies Used

### Frontend

- HTML5
- CSS3
- JavaScript
- Bootstrap 5
- Font Awesome

### Backend

- Flask
- Python

### Machine Learning

- Scikit-Learn
- Pandas
- NumPy
- Joblib

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/omkarsatpute18/Sanjay_Gandhi_Pension_Eligibility_Checker.git
```

---

## Move into Project

```bash
cd Sanjay_Gandhi_Pension_Eligibility_Checker
```

---

## Create Virtual Environment (Optional)

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train the Model

```bash
python main.py
```

---

## Run Flask Server

```bash
python app.py
```

---

# 🖥️ Website

### Live Demo

👉 https://sanjay-gandhi-pension-eligibility-c.vercel.app/

---

# 📸 Screenshots

Example:

```
screenshots/

home.png

prediction.png

result.png
```

---

# 🔄 Working Process

```
Applicant Details
        │
        ▼
Data Preprocessing
        │
        ▼
One Hot Encoding
        │
        ▼
Random Forest Model
        │
        ▼
Prediction
        │
        ▼
Confidence Score
        │
        ▼
Reason Generation
        │
        ▼
Display Result
```

---

# 📈 Prediction Output

The application predicts:

- Eligible
- Not Eligible

Along with

- Confidence Score
- Reason for Prediction

Example

```
Prediction

Eligible

Confidence

97.45%

Reason

Applicant satisfies all eligibility criteria.
```

---

# 📚 Future Improvements

- User Authentication
- Admin Dashboard
- PDF Report Generation
- Database Integration
- Real-time Government API Integration
- Applicant History
- Cloud Deployment
- SMS Notification
- Aadhaar Verification
- Multilingual Support

---

# 🎯 Learning Outcomes

This project demonstrates:

- Machine Learning Model Training
- Data Preprocessing
- One Hot Encoding
- Random Forest Classification
- Model Serialization
- Flask Backend Development
- Responsive Web Design
- Government Portal UI Design
- GitHub Project Management

---

# 📄 License

This project is developed for educational purposes.

Feel free to modify and improve it.

---

# 👨‍💻 Author

**Omkar Satpute**

GitHub:

https://github.com/omkarsatpute18

LinkedIn:

https://www.linkedin.com/in/omkar-satpute-221356349/

---

# ⭐ Support

If you found this project helpful,

⭐ Star the repository

🍴 Fork the project

📢 Share it with others

---

# 🙏 Acknowledgements

- Government of Maharashtra
- Department of Social Justice
- Scikit-Learn
- Flask
- Bootstrap
- Font Awesome
- Open Source Community

---

## 🌟 Project Status

✅ Completed

🚀 Ready for Deployment

💡 Open for Future Enhancements
