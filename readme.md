# Car Price Predictor

This project uses machine learning to predict the resale price of used cars based on historical data. It features a robust data cleaning pipeline and a **Linear Regression** model integrated into a Scikit-Learn **Pipeline**.

---

## Data Cleaning

The notebook performs extensive preprocessing to transform raw data into a usable format:

- **Year**: Filtered for numeric values and converted to integers.
- **Price**: Removed "Ask For Price" entries and converted formatted strings to integers.
- **Kilometers**: Cleaned "kms" units and commas from the `kms_driven` column.
- **Model Names**: Simplified car names to the first three words to categorize brands more effectively.
- **Outliers**: Removed high-end luxury cars (Price > 6,000,000) to prevent model skewing.

---

## Model Pipeline

The project uses an automated pipeline for consistent preprocessing and prediction:

- **Preprocessing**: Utilizes `OneHotEncoder` via `ColumnTransformer` for categorical features (`name`, `company`, `fuel_type`).
- **Algorithm**: Implements `LinearRegression` for price estimation.
- **Performance**: Achieved an **$R^2$ score of ~0.845** by optimizing the train-test split.

---

## Usage

### Prerequisites

```bash
pip install pandas scikit-learn numpy
```
