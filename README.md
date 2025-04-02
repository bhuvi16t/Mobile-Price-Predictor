# Mobile Price Predictor

📌 **Overview**

This project predicts the price of a mobile phone based on its specifications. The dataset was collected from Flipkart using web scraping with BeautifulSoup. The data was then cleaned, transformed, and analyzed using various machine learning techniques. A Random Forest Regressor was selected as the best model, achieving an R² score of 99.8%. Finally, a Streamlit-based web app was developed to allow users to input mobile specifications and get a price prediction.

🚀 **Tech Stack**

* Python
* Web Scraping: BeautifulSoup
* Data Processing & EDA: Pandas, NumPy, Matplotlib, Seaborn
* Machine Learning: Scikit-Learn (Random Forest Regressor)
* Model Deployment: Streamlit

📂 **Folder Structure**

Mobile-Price-Predictor/
├── data collection/            # Scripts for web scraping data from Flipkart
├── app.py                      # Streamlit web app for prediction
├── Entire Workflow Data Cleaning to Prediction.ipynb  # Jupyter Notebook with full workflow
├── mobile_data.xls             # Raw dataset
├── Mobile_tranformed_Data.xls  # Transformed dataset
├── prediction_model/           # Trained model files
│   ├── model.pkl
│   └── ... (other model related files if any)
├── preprocessing_pipeline.pkl  # Preprocessing pipeline
├── price_scaler.pkl            # Price scaler
├── requirements.txt            # List of dependencies
├── README.md                   # Project documentation


📊 **Dataset Collection**

The dataset was collected from Flipkart using BeautifulSoup to scrape mobile phone details such as:

* 📱 Brand
* ⚡ Processor
* 🔹 RAM
* 💾 Storage (ROM)
* 🔋 Battery capacity
* 🖥️ Display type and size
* 📷 Camera specifications
* ⭐ Ratings
* 💰 Price

🔧 **Data Processing**

* ✔ Removed missing values and duplicates
* ✔ Performed Exploratory Data Analysis (EDA) using visualizations
* ✔ Encoded categorical features (Brand, Processor) using One-Hot Encoding
* ✔ Scaled numerical features using StandardScaler
* ✔ Applied feature transformation and selection

🎯 **Model Training & Evaluation**

* ✔ Tried multiple models and selected Random Forest Regressor
* ✔ Achieved 99.8% R² score on the test dataset
* ✔ Saved the trained model and preprocessing pipeline using Joblib

🌐 **Streamlit Web App**

A Streamlit-based web application was developed to allow users to predict mobile prices based on specifications.

🔥 **Features:**

* ✅ Users can enter specifications such as Brand, Processor, RAM, etc.
* ✅ The app preprocesses input data and predicts the price using the trained model.
* ✅ Displays the predicted price dynamically.

🛠 **How to Run the Project**

1️⃣ **Clone the Repository**

```bash
git clone [https://github.com/bhuvi16t/Mobile-Price-Predictor.git](https://github.com/bhuvi16t/Mobile-Price-Predictor.git)
cd Mobile-Price-Predictor
```
**Or Download the repository as a ZIP file:**

[Download ZIP](https://github.com/bhuvi16t/Mobile-Price-Predictor)

**2️⃣ Set Up Virtual Environment**

```Bash

python -m venv myenv
source myenv/bin/activate   # On Mac/Linux
myenv\Scripts\activate     # On Windows
```
**3️⃣ Install Dependencies**

```Bash

pip install -r requirements.txt
```
**4️⃣ Run Streamlit App**

```Bash

streamlit run app.py
```
**🤝 Contributing**

If you would like to contribute to this project, feel free to fork the repository and submit a pull request. Your contributions are always welcome! 🎉   

**📜 License**

This project is licensed under the MIT License.

**👨‍💻 Author**

Bhoopendra Vishwakarma 

Feel free to ⭐ this repository if you found it useful! 😊
