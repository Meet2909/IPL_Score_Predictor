
# 🏏 IPL Win Probability Predictor

[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)](https://share.streamlit.io/) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Model-green?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0id2hpdGUiPjxwYXRoIGQ9Ik0xMi42MywyLjM3Yy0uMzQtLjM0LS44LS41Ny0xLjI5LS41N0gxMy41QzEwLjUsMS44LDkuMDMsMy4zLDguMDcsNS4yMkw4LjA1LDUuMjdMOCw1LjM5bC0uMDUsLS4xMUM3LDEwLjYsNS41LDEyLjIsMi41LDEyLjJINS4yN2MuNDksMCwuOTQtLjIzLDEuMjktLjU3bC4wMS0uMDJjLjM0LS4zNC41Ny0uOC41Ny0xLjI5VjEwLjJMNy40LDguMWwtMS42Ny0xLjY3Yy0uNDItLjQyLS40Mi0xLjEsMC0xLjUybC4wMi0uMDJjLjQyLS40MiwxLjEtLjQyLDEuNTIsMGwxLjY3LDEuNjdMMTAuNiw4LjFsLDEuMjgsMS4yOFY5LjA5YzAsLjQ5LjIzLjk0LjU3LDEuMjlsLjAxLjAxYy4zNC4zNC44LjU3LDEuMjkuNTdoMi43N0MxMi41LDEwLjksMTEsOS40LDkuOTMsNy40N0w5LjkxLDcuNEw5Ljg2LDcuMjlsLjA1LDBDOS45LDcuMjgsMTEuNCw1LjgsMTMuNSw1LjhIMTAuMzdjLS40OSwwLS45NC4yMy0xLjI5LjU3bC0uMDEuMDJjLS4zNC4zNC0uNTcuOC0uNTcsMS4yOVY3Ljk1TDEwLjIsOS4xM2wyLjM4LTIuMzhjLjQyLS40Mi40Mi0xLjEsMC0xLjUybC0uMDItLjAyYy0uNDItLjQyLTEuMS0uNDItMS41MiwwTDEwLjIsNS40N0w5LjEyLDYuN1YxMy41YzAsLjQ5LS4yMy45NC0uNTcsMS4yOWwtLjAxLjAxYy0uMzQuMzQtLjguNTctMS4yOS41N0g0LjVjMywwLDQuNS0xLjUsNS40Ny0zLjQ3TDkuOTksMTEuOUwxMC4wNCwxMS43N2wtLjA1LDAsLjAxLjAzYzEuMDcsMS45MywyLjU5LDMuNDMsNS41LDMuNDNoMi43N2MtLjQ5LDAtLjk0LS4yMy0xLjI5LS41N2wtLjAxLS4wMWMtLjM0LS4zNC0uNTctLjgtLjU3LTEuMjloMFYxMC4yTDExLjEsOC40N2wxLjY3LTEuNjdjLjQyLS40Mi40Mi0xLjEsMC0xLjUybC0uMDItLjAyYy0uNDItLjQyLTEuMS0uNDItMS41MiwwTDEwLjA3LDEuOTVjLjM1LS4zNS44MS0uNTgsMS4zMS0uNThIMy41Yy40OSwwLC45NC4yMywxLjI5LjU3bC4wMS4wMmMuMzQuMzQuNTcuOC41NywxLjI5djEuMjVMNCw1LjgyTDIsMy44NGMtLjQyLS40Mi0uNDItMS4xLDAtMS41MmwuMDItLjAyYy40Mi0uNDIsMS4xLS40MiwxLjUyLDBMNS4yMSwzLjk4bDQuMzktNC4zOWMuMzUtLjM1LjgyLS41OCwxLjMyLS41OEg0LjVDNy41LDIsOS4wMywzLjUsMTAuMDYsNS40N2wuMDIsLjA1bC4wNS4xMS0uMDUtLjExQzkuMDMsMTAuNSw3LjUsMTIuMSw0LjUsMTIuMWgtMi43N2MuNDksMCwuOTQtLjIzLDEuMjktLjU3bC4wMS0uMDJjLjM0LS4zNC41Ny0uOC41Ny0xLjJWMTAuMUw1LjMzLDguNDdsMS42Ny0xLjY3Yy40Mi0uNDIsLjQ2LTEuMDksLjA4LTEuNTRaIi8+PC9zdmc+)]

This project predicts the win probability for the chasing team in an IPL (Indian Premier League) match in real-time. It uses an **XGBoost machine learning model** trained on historical ball-by-ball data and is deployed as an interactive **Streamlit** web application.

---

## 🌟 Key Features

* **📈 Live Win Probability:** Input the current match state (teams, venue, target, score, overs, etc.) to get an instant win probability for the chasing team.
* **📊 Historical Match Analysis:** Select any two teams and choose a specific historical match to visualize the ball-by-ball chase.
* **🚀 Interactive Visualizations:**
    * **Win Probability Progression:** A line chart showing how the win percentage changed with every ball.
    * **Run Rate Progression:** A "worm graph" comparing the chasing team's current run rate (CRR) against the required run rate (RRR).
    * **Wickets Fallen Over Time:** A step plot visualizing when each wicket fell.
* **🧠 Advanced Feature Engineering:** The model's accuracy is enhanced by using features like `runs_last_5_overs`, `wickets_last_5_overs`, `current_run_rate`, and `required_run_rate`.

---

## 📸 Application Demo

<img width="2559" height="870" alt="image" src="https://github.com/user-attachments/assets/2b38a671-d083-40eb-9910-2138116b6925" />

<img width="2559" height="1288" alt="image" src="https://github.com/user-attachments/assets/b3112fe6-f69d-4079-a40d-25af6bb85a20" />

<img width="2559" height="1344" alt="image" src="https://github.com/user-attachments/assets/d2fc9519-f41a-48c6-b4c7-bc40fd929cd7" />

<img width="2559" height="924" alt="image" src="https://github.com/user-attachments/assets/b777fce9-d9f0-48ee-9b7f-d0a2ae494b4f" />



### 1. Custom Scenario Predictor
This mode allows you to input any match scenario to get a live prediction.

### 2. Historical Match Analysis
This mode lets you load a past match and see how the win probability, run rate, and wickets changed throughout the second innings.

---

## 🛠️ Tech Stack

* **Machine Learning:** Scikit-learn, XGBoost, Pandas, NumPy
* **Web Framework:** Streamlit
* **Data Visualization:** Plotly
* **Model Persistence:** Joblib

---

## 📦 Project Structure

```
IPL_Score_Predictor/
├── .git/
├── data/
│   ├── matches.csv         # Raw match data (required)
│   └── deliveries.csv      # Raw ball-by-ball data (required)
│
├── .gitignore
├── app.py                      # The Streamlit web application script
├── datapreprocessing(more accurate).ipynb  # Jupyter Notebook for data processing and model training
├── xgb_ipl_predictor.joblib    # (Generated) The trained XGBoost model
├── column_transformer.joblib   # (Generated) The Scikit-learn ColumnTransformer
├── analysis_data.joblib      # (Generated) Processed data for historical analysis
├── README.md
└── requirements.txt            # Python dependencies
```

---

## 🚀 How to Run This Project Locally

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites
* Python 3.10 or newer
* Git

### 2. Download the Data
The model is trained on the **IPL Complete Dataset (2008-2022)** from Kaggle.
1.  Download the dataset from [this Kaggle link](https://www.kaggle.com/datasets/patrickb1912/ipl-complete-dataset-20082020).
2.  You will need two files: `matches.csv` and `deliveries.csv`.

### 3. Clone the Repository
```bash
git clone [https://github.com/Meet2909/IPL_Score_Predictor.git](https://github.com/Meet2909/IPL_Score_Predictor.git)
cd IPL_Score_Predictor
```

### 4. Set Up the Environment
It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 5. Install Dependencies
A `requirements.txt` file is included with all necessary libraries.

```bash
pip install -r requirements.txt
```

### 6. Place the Data
Create a `data` folder in the root of the project and place the downloaded `matches.csv` and `deliveries.csv` files inside it.

```
IPL_Score_Predictor/
├── data/
│   ├── matches.csv
│   └── deliveries.csv
├── app.py
...
```

### 7. Train the Model
You must run the data preprocessing and training notebook **first**. This notebook generates the `*.joblib` files that the Streamlit app depends on.

1.  Ensure you have `jupyter` installed (`pip install jupyterlab`).
2.  Run the notebook:
    ```bash
    jupyter-lab "datapreprocessing(more accurate).ipynb"
    ```
3.  Inside Jupyter Lab, click **"Run" > "Run All Cells"**.
4.  This will create three new files in your root directory:
    * `xgb_ipl_predictor.joblib`
    * `column_transformer.joblib`
    * `analysis_data.joblib`

### 8. Run the Streamlit App
Once the model files are generated, you can start the web application.

```bash
streamlit run app.py
```

Your browser should automatically open to the application's local address (usually `http://localhost:8501`).

---

## 🤖 The Model Explained

### Data Source
The model is trained on ball-by-ball data from the IPL (2008-2022).

### Model
An **XGBoost Classifier** (`xgb.XGBClassifier`) is used. This is a powerful gradient-boosting algorithm well-suited for tabular data and classification tasks. The model's hyperparameters were optimized using `RandomizedSearchCV` to find the best-performing combination.

### Key Features
The model's prediction is based on the following features, which are engineered from the raw data:

* **Categorical Features:**
    * `venue`
    * `batting_team`
    * `bowling_team`
* **Numerical Features:**
    * `runs_to_chase` (The target score)
    * `current_score` (Runs scored by the chasing team)
    * `wickets_remaining` (10 - wickets lost)
    * `balls_left` (120 - balls bowled)
    * `current_run_rate` (CRR)
    * `required_run_rate` (RRR)
    * `runs_last_5_overs` (Runs scored in the previous 30 balls)
    * `wickets_last_5_overs` (Wickets lost in the previous 30 balls)

The categorical features are one-hot encoded using a `ColumnTransformer`, which is saved and loaded by the app to ensure consistent processing for live predictions.

---

## Future Improvements

* Deploy the Streamlit app to a public-facing service (e.g., Streamlit Community Cloud).
* Add more advanced plots, such as player-specific contributions.
* Retrain the model with data from newer IPL seasons (2023+).

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
