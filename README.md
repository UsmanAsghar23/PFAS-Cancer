# PFAS and Cancer Trends in California

This project analyzes cancer trends in California with a focus on the potential impact of water quality—specifically PFAS contamination—on public health. It investigates whether higher PFAS levels are associated with increased cancer rates, especially in low-income communities.

## 🧪 Data Sources

1. **California Cancer Dataset (2017–2021)**  
   County-level cancer incidence by type, ethnicity, gender, and age-adjusted values.

2. **California PFAS Dataset (2014–2016)**  
   Over 680,000 groundwater samples with PFAS concentration and water quality metrics.

3. **California County Shapefiles (2022)**  
   Geographical boundaries used to spatially link cancer and water data.


## 💡 Big Idea

Water contaminants may impact cancer rates, but the link is still unclear.

### 🎯 Project Goal:
Analyze California’s water quality and health disparities to explore whether there’s a correlation between PFAS (forever chemicals) contamination and increased cancer incidence.

### ❓ Why This Problem?
- Cancer rates are rising.
- PFAS contamination in water is becoming more widespread.
- More studies are emerging on PFAS–cancer links.
- We aim to provide evidence-based insights for further action.

## 📁 Repository Contents

| File/Folder | Description |
|-------------|-------------|
| `data_cleaning/` | Directory for cleaning scripts and data preparation utilities. |
| `cancer_data_cleaning.py` | Script for cleaning cancer incidence data. |
| `cleaned_pfas_cancer_merged.csv` | Merged dataset combining PFAS and cancer data. |
| `eda_cancer_vs_pfas.ipynb` | Exploratory data analysis between cancer rates and PFAS. |
| `feature.ipynb` | Feature engineering and preprocessing notebook. |
| `final-report.ipynb` | Final report notebook with visuals and conclusions. |
| `Individual Contributions for Progress Report.pdf` | Team contributions for progress report. |
| `Individual Contributions for Project.pdf` | Final project contribution breakdown. |
| `ml_analysis.ipynb` | Machine learning experiments and model evaluations. |
| `ml.py` | Script for training/testing models on cleaned datasets. |
| `pfas_data_cleaner.py` | Script for cleaning PFAS water contamination data. |
| `Progress-report.ipynb` | Notebook tracking project timeline and deliverables. |
| `README.md` | This file — overview of the project and its structure. |
| `reduced_pfas_dataset.csv` | Filtered PFAS dataset used in early analysis. |
| `reduced_pfas_dataset1.csv` | Another version of reduced PFAS dataset. |
| `requirements.txt` | Python dependencies needed to run the project. |
| `visualizations.py` | Script for generating final graphs and maps. |

## 🛠 Technologies Used

- Python (Pandas, NumPy, GeoPandas)
- Jupyter Notebooks
- Matplotlib, Seaborn,
- Scikit-learn

## 🚀 How to Use

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/pfas-cancer-california.git
   cd pfas-cancer-california
2. Create a virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
3. Launch Jupyter Notebook and begin analysis:


## 👥 Contributors

| Name              | GitHub Username     |
|-------------------|---------------------|
| Usman Asghar      |[UsmanAsghar23](https://github.com/UsmanAsghar23) |
| Shayan Khan       |[mkhan405](https://github.com/mkhan405)         |
| Omar Khan         |[oKhan0](https://github.com/okhan0) 
| Zuhayr Saeed      |[zsaee2](https://github.com/zsaee2)             |
| Hamza Shaikh      |[Hamza-developer1](https://github.com/Hamza-developer1) 

## ✅ Actual Repository With All Commits
https://github.com/uic-cs418/group-project-data-engineers#



