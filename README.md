# NYC Airbnb Price Prediction (Linear Regression)

Exploring the 2019 New York City Airbnb listings and modelling nightly price with linear regression.

## Question
What drives the nightly price of an Airbnb listing in NYC, and how well can a simple linear model predict it?

## Data
`AB_NYC_2019.csv` comes from Inside Airbnb, via Kaggle. It has 48,895 listings and 16 columns, including:
- location (borough, neighbourhood, latitude/longitude)
- room type
- price
- minimum nights
- review counts and frequency
- host listing count
- availability

Prices are very skewed: the median is **$106**, while the maximum is **$10,000**.

## Approach
1. **Cleaning:**
   - checked for duplicates (none)
   - filled missing names
   - set missing `reviews_per_month` to 0
2. **EDA:**
   - listings per borough
   - Kendall correlation heatmap
   - map of listings coloured by availability
   - pair plot of numeric features
3. **Encoding:** label-encoded borough, neighbourhood and room type.
4. **Model:** `LinearRegression` using neighbourhood, room type, minimum nights, host listing count and availability, with a 70/30 train/test split. A `statsmodels` OLS summary is included for coefficient significance.

## Results
- Test R² is **0.09**: the current features explain very little of the variation in price.
- Room type is by far the strongest predictor. Entire homes cost about $110/night more than private rooms.

## How to run
```bash
pip install numpy pandas scikit-learn statsmodels matplotlib seaborn
jupyter notebook Airbnb_LR.ipynb
```

## Limitations and next steps
- `neighbourhood` (220+ categories) is label-encoded and treated as a number, so the model assumes neighbourhood 150 is "more" than neighbourhood 20. Use one-hot encoding or target encoding instead.
- Price should be log-transformed, and extreme outliers ($0 and $10,000 listings) removed, before fitting a linear model. The model currently predicts some negative prices.
- R² is labelled "accuracy" in the notebook; it should be reported as R², alongside MAE in dollars.
- Next: add latitude/longitude or distance to Midtown, reviews and text features from the listing name, then compare with a gradient-boosted model.
