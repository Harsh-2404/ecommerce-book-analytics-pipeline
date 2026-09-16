# 📚 E-Commerce Book Analytics Pipeline

An end-to-end data engineering, analytics, and machine learning pipeline built on scraped e-commerce book catalog data. This repository features automated web scraping, data cleaning, exploratory data analysis (EDA), predictive pricing models, an interactive Power BI dashboard, and an executive project report.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Pipeline Architecture](#-pipeline-architecture)
- [Key Insights & Findings](#-key-insights--findings)
- [Power BI Dashboard Preview](#-power-bi-dashboard-preview)
- [Machine Learning Performance](#-machine-learning-performance)
- [Installation & Setup](#-installation--setup)
- [License](#-license)

---

## 📖 Project Overview
This project extracts book catalog information from an e-commerce platform to understand market trends, pricing strategies, rating distributions, and genre popularity.

**Core Objectives:**
1. **Web Scraping:** Extract structured catalog data (Title, Price, Rating, Category, Availability).
2. **Data Cleaning & Preprocessing:** Handle missing values, parse data types, and prepare analytical datasets.
3. **Exploratory Data Analysis (EDA):** Identify key market trends, high-value genres, and rating correlates.
4. **Predictive Modeling:** Train Machine Learning regression models to predict book prices based on features.
5. **Business Intelligence:** Design an interactive Power BI dashboard for business stakeholders.

---

## 📁 Repository Structure

```text
ecommerce-book-analytics-pipeline/
├── data/
│   ├── raw_books_data.csv            # Raw scraped dataset
│   └── clean_books_data.csv          # Preprocessed analytical dataset
├── notebooks/
│   ├── 01_web_scraping.ipynb         # Web scraping scripts (BeautifulSoup/Requests)
│   ├── 02_data_cleaning.ipynb        # Data parsing & cleaning workflow
│   ├── 03_eda_visualization.ipynb    # Statistical analysis & visualizations
│   └── 04_model_building.ipynb       # ML regression algorithms & evaluation
├── dashboard/
│   └── 5_Books_Analytics_Dashboard.pbix # Power BI interactive dashboard file
├── report/
│   └── Harsh_Srivastav_Capstone_Project_Report.pdf # Executive summary & final report
├── .gitignore                        # Standard Python gitignore
├── LICENSE                           # MIT License
└── README.md                         # Project documentation
```
---

## ⚡ Pipeline Architecture
```text
[Web Scraping] ──> [Raw CSV] ──> [Cleaning & Wrangling] ──> [Clean CSV]
                                                               │
        ┌──────────────────────────────────────────────────────┴──────────────────────────────────────────────────────┐
        │                                                      │                                                      │
        ▼                                                      ▼                                                      ▼
[EDA & Visualizations]                              [Machine Learning Models]                              [Power BI Dashboard]
(Notebooks 03)                                      (Notebook 04 - Regression)                             (Interactive Analytics)

```
---
## 📊 Key Insights & Findings
- **Pricing Strategy:** Non-fiction and technical categories command higher average prices compared to fiction.
- **Stock & Rating Correlation:** Higher-rated books (4+ stars) maintain more consistent inventory levels.
- **Top Performing Genres:** Fiction, Business, and Science represent over 40% of the catalog volume.
---

## 🖥️ Power BI Dashboard Preview
The interactive dashboard (dashboard/5_Books_Analytics_Dashboard.pbix) provides key metrics including:
- Total Revenue & Inventory Breakdown
- Category-wise Average Pricing & Rating Distribution
- Stock Availability vs. Book Popularity Matrix

(To view the dashboard, open the .pbix file in Power BI Desktop or download it from the dashboard/ directory).

---

## 🤖 Machine Learning Performance
Multiple regression algorithms were evaluated to predict book prices based on categorical and numerical features:

| Model | MAE | RMSE | R² Score |
| :--- | :---: | :---: | :---: |
| Linear Regression | 4.21 | 5.34 | 0.68 |
| Random Forest Regressor | **2.15** | **3.02** | **0.84** |
| XGBoost Regressor | 2.30 | 3.18 | 0.82 |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### Local Environment Setup
```bash
# 1. Clone the repository
git clone [https://github.com/Harsh-2404/ecommerce-book-analytics-pipeline.git](https://github.com/Harsh-2404/ecommerce-book-analytics-pipeline.git)

# 2. Navigate to project directory
cd ecommerce-book-analytics-pipeline

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install required dependencies
pip install pandas numpy matplotlib seaborn scikit-learn beautifulsoup4 requests
```
---

## 📜 License
Distributed under the **MIT License**. See LICENSE for more information.
