🌍 Climate Data Visual Explorer

An AI-powered interactive web app built with Streamlit that provides:

Real-time weather data 🌤️

Historical climate trend analysis 📈

Future forecasts 🔮

Anomaly detection ⚠️

Insights & summaries 📝

This project combines OpenWeatherMap API, machine learning models (Random Forest), and data visualization to give users a smart climate dashboard.

✨ Features

✅ Current Weather – Temperature, humidity, pressure, wind speed & conditions
✅ Historical Trends – Temperature & humidity variations with visual charts
✅ Forecasting – Predicts next 5 hours of temperature & humidity
✅ Anomaly Detection – Detects unusual weather patterns in historical data
✅ Insights Summary – Highlights averages, extremes, and seasonal patterns

🛠️ Tech Stack

Python 3.9+

Streamlit – interactive UI

scikit-learn – ML models (Random Forest, Isolation Forest)

Pandas / NumPy – data handling

Matplotlib / Seaborn – visualizations

OpenWeatherMap API – live weather data

📦 Installation

Clone the repo

git clone https://github.com/your-username/climate-visual-explorer.git
cd climate-visual-explorer


Create a virtual environment (recommended)

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\\Scripts\\activate    # Windows


Install dependencies

pip install -r requirements.txt


Add your OpenWeatherMap API key
In the code, replace API_KEY with your own key from OpenWeatherMap
.

▶️ Run the App
streamlit run climate.py


Then open http://localhost:8501
 in your browser 🎉

📊 Screenshots
🌍 Current Weather

(screenshot placeholder)

📈 Historical Trends

(screenshot placeholder)

🔮 Forecasts

(screenshot placeholder)

📌 Project Structure
📂 climate-visual-explorer
│── climate.py          # Main Streamlit app
│── weather.csv         # Historical dataset
│── requirements.txt    # Dependencies
│── README.md           # Documentation

🚀 Future Improvements

🌐 Add multi-city comparison

🎨 Dark mode & custom theming

📅 Weekly & monthly forecast extension

📍 Geo-location based weather fetch

🤝 Contributing

Pull requests are welcome! If you’d like to add new features or fix issues, fork the repo and submit a PR.

📜 License

This project is licensed under the MIT License – feel free to use and modify.
