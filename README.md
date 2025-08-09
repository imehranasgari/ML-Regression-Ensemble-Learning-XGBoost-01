# CO2 Emissions Prediction with XGBoost

## Problem Statement and Goal

This project aims to predict carbon dioxide (CO2) emissions from coal-based electricity production in the United States using the XGBoost regression algorithm. The dataset spans 1973 to 2016, and the goal is to forecast future emissions based on temporal features. This project demonstrates my proficiency in building a regression pipeline, including data preprocessing, feature engineering, model training, and evaluation, showcasing my ability to tackle real-world time-series problems for a professional portfolio.

## Solution Approach

The project follows a structured machine learning workflow:
- **Data Preprocessing**: Loaded the dataset, extracted year and month from the `YYYYMM` column, and handled missing values using mean imputation.
- **Feature Engineering**: Created temporal features (`Year`, `Month`) to capture trends and seasonality.
- **Model Training**: Trained an XGBoost Regressor to predict CO2 emissions, leveraging its gradient boosting capabilities.
- **Hyperparameter Tuning**: Used GridSearchCV to optimize XGBoost parameters for improved accuracy.
- **Performance Evaluation**: Assessed the model using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² Score.
- **Visualization**: Plotted actual vs. predicted emissions and forecasted future emissions to visualize model performance.

## Technologies & Libraries

- **Python 3.9+**: Core programming language
- **pandas**, **numpy**: Data manipulation and analysis
- **xgboost**: Gradient boosting for regression
- **scikit-learn**: Train-test split, evaluation metrics, and hyperparameter tuning
- **matplotlib**, **seaborn**: Data visualization

## Description about Dataset

The dataset contains monthly CO2 emissions from coal-based electricity production in the United States (1973–2016). It includes:
- **YYYYMM**: Date in the format YYYYMM (e.g., 197301 for January 1973).
- **Value**: CO2 emissions in million metric tons of carbon dioxide.

The dataset is sourced from a public repository (not specified in the notebook) and contains 528 records (44 years × 12 months). Missing values, if any, are handled during preprocessing.

## Installation & Execution Guide

### Prerequisites
- Python 3.9+
- Jupyter Notebook
- Dataset: `co2_emissions.csv` (a CSV file with `YYYYMM` and `Value` columns)

### Installation
Install required packages using pip:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Execution
1. Ensure the `co2_emissions.csv` file is in the same directory as the notebook.
2. Run the notebook:
   ```bash
   jupyter notebook "XGBoost (Regression).ipynb"
   ```
3. The notebook generates visualizations and saves predictions in `predictions.csv`.

## Key Results / Performance

- **Model**: XGBoost Regressor with tuned hyperparameters.
- **Evaluation Metrics**: 
  - Root Mean Squared Error (RMSE): 4.263376
  - R² Score: 0.990117
  - **Cross-Validation**: 0.98
- **Output**: Predicted CO2 emissions for 2016–2018 were saved in `predictions.csv`.

Detailed metrics are available in the notebook output.

## Screenshots / Sample Outputs

- **Actual vs. Predicted Emissions Plot**: A line plot comparing actual and predicted CO2 emissions on the test set.
- **Forecast Plot**: A line plot showing forecasted emissions from August 2016 onward.

> 💡 *Some interactive outputs (e.g., plots) may not display correctly on GitHub. Please view this notebook via [nbviewer.org](https://nbviewer.org) for full rendering.*

## Additional Learnings / Reflections

This project enhanced my skills in:
- Applying XGBoost for regression tasks, leveraging its efficiency and scalability.
- Feature engineering for time-series data, such as extracting temporal components.
- Handling and evaluating regression models with appropriate metrics (MSE, RMSE, R²).
- Visualizing time-series predictions to communicate results effectively.

The project highlighted the importance of temporal feature engineering and hyperparameter tuning in achieving accurate predictions for time-series data.

## 👤 Author

**Mehran Asgari**  
**Email**: [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com)  
**GitHub**: [https://github.com/imehranasgari](https://github.com/imehranasgari)

## 📄 License

This project is licensed under the MIT License – see the `LICENSE` file for details.