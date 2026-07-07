<div align="center">

# 🏙️ CityMatch
### AI-Powered Delhi Neighbourhood Recommendation System

Find the best locality in Delhi based on **Air Quality, Traffic, Rental Prices, and Amenities** using Machine Learning.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

# 📌 Overview

CityMatch is an AI-powered neighbourhood recommendation system designed to help students, professionals, and families choose the most suitable place to live in Delhi.

Unlike traditional property websites that only compare prices, CityMatch combines multiple real-world datasets to generate an intelligent neighbourhood score using Machine Learning.

The system analyzes:

- 🌫️ Air Quality (PM2.5 AQI)
- 🚗 Traffic Congestion
- 🏠 Rental Prices
- 🏥 Nearby Amenities
- 🤖 Machine Learning Clustering
- 📊 Interactive Dashboard

---

# 🎯 Problem Statement

Choosing a place to live is difficult because users must compare multiple factors manually.

Most real estate platforms focus only on:

- Property prices
- Basic location information

They ignore important quality-of-life indicators such as pollution, traffic, and accessibility.

CityMatch solves this by combining multiple urban datasets into a single recommendation engine.

---

# ✨ Features

- ✅ AI-powered neighbourhood recommendation
- ✅ Real-time Air Quality integration
- ✅ Traffic Congestion analysis
- ✅ Rental affordability comparison
- ✅ Amenity availability analysis
- ✅ Machine Learning clustering
- ✅ Weighted scoring algorithm
- ✅ Interactive Streamlit dashboard
- ✅ Data visualization with Plotly
- ✅ Easy-to-use interface

---

# 🧠 Machine Learning Pipeline

```
Raw Data
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
Normalization
(MinMaxScaler)
     │
     ▼
K-Means Clustering
(K = 4)
     │
     ▼
Weighted Scoring Engine
     │
     ▼
Neighbourhood Ranking
     │
     ▼
Interactive Dashboard
```

---

# 📊 Data Sources

| Dataset | Source |
|----------|--------|
| Air Quality (PM2.5 AQI) | CPCB |
| Traffic Congestion | TomTom Flow API |
| Rental Prices | Kaggle |
| Amenities | OpenStreetMap (OSMnx) |

---

# ⚙️ Tech Stack

## Programming

- Python

## Machine Learning

- Scikit-learn
- K-Means Clustering
- MinMaxScaler

## Data Processing

- Pandas
- NumPy

## Visualization

- Plotly
- Matplotlib

## Dashboard

- Streamlit

## APIs

- CPCB AQI
- TomTom Flow API
- OpenStreetMap (OSMnx)

---

# 📂 Project Structure

```
CityMatch/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│
├── models/
│
├── app.py
├── requirements.txt
├── README.md
└── assets/
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/aakarshpab/Citymatch.git
```

Move into the project directory

```bash
cd Citymatch
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 📈 Recommendation Methodology

Each locality receives a final score using weighted factors.

Example:

| Parameter | Weight |
|-----------|---------|
| Air Quality | 30% |
| Traffic | 25% |
| Rental Cost | 25% |
| Amenities | 20% |

The final score determines the neighbourhood ranking.

---

# 📊 Machine Learning

Model Used:

- K-Means Clustering

Number of Clusters:

```
K = 4
```

Evaluation Metric

- Silhouette Score

Feature Scaling

- MinMaxScaler

---

# 📸 Screenshots

> Add your screenshots here.

Example:

```
assets/homepage.png
assets/dashboard.png
assets/clusters.png
```

---

# 🔮 Future Improvements

- Live traffic updates
- Crime rate integration
- Metro accessibility score
- Weather impact analysis
- Personalized user preferences
- Deep Learning recommendation engine
- Mobile application

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch

```
git checkout -b feature-name
```

3. Commit changes

```
git commit -m "Added new feature"
```

4. Push

```
git push origin feature-name
```

5. Open a Pull Request

---

# 👨‍💻 Author

### Aakarsh Pab

Final Year B.Tech (Computer Science)

GitHub:
https://github.com/aakarshpab

---

# 📜 License

This project is licensed under the MIT License.

---

# ⭐ Support

If you found this project helpful:

⭐ Star this repository

🍴 Fork it

📢 Share it with others

---

<div align="center">

### Thanks for visiting CityMatch ❤️

Built with Python • Machine Learning • Streamlit

</div>
