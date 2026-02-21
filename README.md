# Stock Price Prediction

A comprehensive machine learning project for predicting stock prices using multiple deep learning models and technical indicators. This project demonstrates the implementation of Linear Regression, LSTM (Long Short-Term Memory), and advanced neural networks with technical analysis features.

## Overview

This project implements multiple approaches to stock price prediction:

1. **Linear Regression Baseline** - Simple linear regression model to establish a baseline
2. **Basic LSTM** - Single LSTM model with 60-day time steps for temporal pattern learning
3. **Advanced LSTM with Technical Indicators** - Multi-feature LSTM incorporating RSI, MACD, and other technical indicators

## Features

### Data Collection
- Uses Yahoo Finance (`yfinance`) to download historical stock data
- Default dataset: Apple (AAPL) stock prices from 2015-2024
- Includes OHLCV (Open, High, Low, Close, Volume) data

### Technical Indicators
- **RSI (Relative Strength Index)** - Momentum oscillator measuring magnitude of price changes
- **MACD (Moving Average Convergence Divergence)** - Trend-following momentum indicator
- **Signal Line** - 9-period EMA of MACD for trading signals

### Models Implemented

#### 1. Linear Regression Model
- Quick baseline for comparison
- RMSE: ~21.86 on test set
- Limitations: Cannot capture non-linear patterns in stock prices

#### 2. Basic LSTM Model
- Single layer LSTM with 50 units
- Time step: 60 days
- Outputs: Single LSTM layer with 10 epochs training
- Better captures temporal dependencies than linear models

#### 3. Advanced LSTM with Multi-Feature Input
- Two LSTM layers with 64 units each
- Input features: Open, High, Low, Close, Volume, RSI, MACD, MACD Signal (8 features)
- Time step: 60 days
- **RMSE: ~2.73** (significant improvement)
- More robust and generalizable predictions

## Installation

```bash
# Install required packages
pip install yfinance pandas numpy scikit-learn tensorflow matplotlib ta
```

## Dependencies

- **yfinance** (0.2.66+) - Download financial data from Yahoo Finance
- **pandas** (2.2.2+) - Data manipulation and analysis
- **numpy** (2.0.2+) - Numerical computations
- **scikit-learn** - Machine learning utilities (preprocessing, metrics)
- **tensorflow/keras** (2.19.0+) - Deep learning framework for LSTM models
- **matplotlib** - Data visualization
- **ta** (0.11.0+) - Technical analysis indicators

## Project Structure

```
Stock_Price-Prediction/
├── Stock_Price_Prediction.ipynb   # Main jupyter notebook with all implementations
├── README.md                       # This file
└── .gitignore                      # Git ignore file
```

## Usage

### Running the Jupyter Notebook

```bash
jupyter notebook Stock_Price_Prediction.ipynb
```

### Basic Workflow

1. **Data Download**: Uses `yfinance.download()` to fetch AAPL stock data
2. **Data Preprocessing**: MinMax scaling to normalize features to [0, 1] range
3. **Sequence Creation**: Creates overlapping sequences of 60-day windows for LSTM input
4. **Model Training**: Trains LSTM models with 10 epochs and batch size of 32
5. **Evaluation**: Compares predicted vs actual prices using visualization and RMSE metric

## Model Performance

| Model | RMSE | Advantages | Limitations |
|-------|------|------------|-------------|
| Linear Regression | 21.86 | Fast, interpretable | Cannot capture temporal patterns |
| Basic LSTM | - | Captures sequential patterns | Limited to price data only |
| Advanced LSTM | 2.73 | Multi-feature input, best performance | More computationally expensive |

## Key Insights

1. **Feature Scaling**: MinMax scaling crucial for LSTM convergence
2. **Time Window Size**: 60-day window chosen as balance between computational cost and pattern capture
3. **Technical Indicators**: Adding RSI and MACD significantly improves prediction accuracy
4. **Architecture**: Two-layer LSTM with 64 units provides good balance between capacity and overfitting prevention

## Visualization

The notebook includes visualizations for:
- Historical closing prices with trend analysis
- Linear regression predictions vs actual prices
- LSTM predictions with 30-day future forecasting
- Technical indicators (RSI, MACD) over time
- Comparison between actual and predicted prices

## Future Improvements

- [ ] Add more technical indicators (Bollinger Bands, Stochastic Oscillator)
- [ ] Implement ensemble methods combining multiple models
- [ ] Add validation set for better hyperparameter tuning
- [ ] Implement early stopping to prevent overfitting
- [ ] Support for multiple stock symbols
- [ ] Real-time prediction pipeline
- [ ] Trading strategy implementation based on predictions
- [ ] Attention mechanisms for better temporal pattern capture

## Disclaimer

⚠️ **Important**: This project is for educational purposes only. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always conduct thorough research and consult with financial professionals before making investment decisions.

## Requirements

- Python 3.8+
- GPU recommended for faster training (NVIDIA CUDA compatible)

## Author

Ritvik (@ritvikvr)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a pull request

## References

- [Yahoo Finance API](https://finance.yahoo.com/)
- [LSTM Neural Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Technical Analysis Library (TA)](https://github.com/bukosabino/ta)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/guide/keras)
