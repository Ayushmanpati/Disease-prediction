
# 🩺 Medical Predictions Web App

A **web-based application** built with **Streamlit** to predict the likelihood of **COVID-19**, **Diabetes**, and **Heart Disease** using **machine learning** models. The app uses `RandomForestClassifier` from Scikit-learn for predictions and **Plotly** for interactive data visualizations.

---

## ✨ Features

- **Interactive UI**: Input health metrics via sliders for real-time predictions.
- **Three Prediction Models**:
  - **COVID-19**: Based on symptoms like dry cough, fever, sore throat, and breathing difficulty.
  - **Diabetes**: Uses glucose, insulin, BMI, and age.
  - **Heart Disease**: Considers chest pain, blood pressure, cholesterol, and max heart rate.
- **Data Visualizations**: Interactive scatter plots explore relationships between features and outcomes.
- **Responsive Design**: Custom CSS for a polished, user-friendly interface.

---

## 🛠 Tech Stack

- **Python**: Core programming language
- **Streamlit**: Web app framework
- **Scikit-learn**: For training `RandomForestClassifier` models
- **Pandas & NumPy**: For data manipulation
- **Plotly**: For interactive visualizations
- **CSS**: For custom UI styling

---

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ayushman pati/disease_prediction.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd medical-predictions-app
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Datasets

The app uses three CSV files for training:

- `Covid-19 Predictions.csv`
- `Diabetes Predictions.csv`
- `Heart Disease Predictions.csv`

> ⚠️ Ensure these files are placed in the project directory. Due to privacy and size constraints, they are **not included** in the repository. You may generate or use similar public datasets.

---

## 🧭 Usage

- Launch the app and use the sidebar to navigate between:
  - **Home**
  - **COVID-19**
  - **Diabetes**
  - **Heart Disease**
  - **Plots**
- Use sliders to input health data and get instant predictions.
- Explore scatter plots in the **Plots** section to visualize trends.

---

## 🔮 Future Improvements

- Add more advanced models like **XGBoost** or **Neural Networks**
- Integrate real-time data APIs for up-to-date health metrics
- Improve visualizations with more chart types
- Deploy to cloud platforms like **Heroku** or **Streamlit Cloud**

---

## 🤝 Contributing

Contributions are welcome!  
Fork the repository and submit a pull request with your improvements.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📬 Contact

For questions or collaboration, feel free to reach out via **[LinkedIn](#)** or **GitHub**.

---

**Built with ❤️ using Streamlit | © 2025**

---
