import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry

cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 48.8534,
    "longitude": 2.3488,
    "start_date": "2024-10-19",
    "end_date": "2024-12-20",
    "daily": ["temperature_2m_mean", "precipitation_sum", "rain_sum", "snowfall_sum", "sunrise", "sunset"],
    "timezone": "Europe/London",
}
responses = openmeteo.weather_api(url, params=params)

response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

daily = response.Daily()
daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
daily_precipitation_sum = daily.Variables(1).ValuesAsNumpy()
daily_rain_sum = daily.Variables(2).ValuesAsNumpy()
daily_snowfall_sum = daily.Variables(3).ValuesAsNumpy()
daily_sunrise = daily.Variables(4).ValuesInt64AsNumpy()
daily_sunset = daily.Variables(5).ValuesInt64AsNumpy()

daily_data = {"date": pd.date_range(
    start=pd.to_datetime(daily.Time(), unit="s", utc=True),
    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=daily.Interval()), inclusive="left"),
    "temperature_2m_mean": daily_temperature_2m_mean,
    "precipitation_sum": daily_precipitation_sum,
    "rain_sum": daily_rain_sum,
    "snowfall_sum": daily_snowfall_sum,
    "sunrise": daily_sunrise,
    "sunset": daily_sunset
}

daily_dataframe = pd.DataFrame(data=daily_data)

daily_dataframe.to_csv("../data/meteo/daily_weather_2024_10_19_to_2024_12_20.csv", index=False)