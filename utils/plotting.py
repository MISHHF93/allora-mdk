import matplotlib.pyplot as plt

def plot_forecast(data, forecast, title="Forecast"):
    """
    Plot the original data and forecast.

    Parameters:
    - data (pd.DataFrame): Original historical data with 'date' and 'close' columns.
    - forecast (pd.DataFrame, pd.Series, or np.ndarray): Forecasted values.
    - title (str): Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data['date'], data['close'], label='Actual', linewidth=2)

    # Align forecast with dates from the original data
    forecast_values = forecast.squeeze()  # ensures 1D
    forecast_length = len(forecast_values)
    forecast_dates = data['date'].iloc[-forecast_length:]  # match forecast to most recent dates

    plt.plot(forecast_dates, forecast_values, label='Forecast', linestyle='--')
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
