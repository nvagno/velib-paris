if __name__ == '__main__':
    import openmeteo_requests

    import pandas as pd
    import requests_cache
    from retry_requests import retry

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 48.8534,
        "longitude": 2.3488,
        "start_date": "2025-10-15",
        "end_date": "2025-12-23",
        "daily": ["precipitation_sum", "rain_sum", "snowfall_sum", "sunrise", "sunset"],
        "hourly": ["temperature_2m", "precipitation", "rain", "snowfall", "relative_humidity_2m"],
        "timezone": "Europe/London",
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(4).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    ), "temperature_2m": hourly_temperature_2m, "precipitation": hourly_precipitation, "rain": hourly_rain,
        "snowfall": hourly_snowfall, "relative_humidity_2m": hourly_relative_humidity_2m}

    hourly_dataframe = pd.DataFrame(data=hourly_data)

    hourly_dataframe.to_csv("../data/meteo/daily_weather_2025_12_15_to_2025_12_23.csv", index=False)
