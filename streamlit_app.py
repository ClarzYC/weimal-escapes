import altair as alt
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
from pprint import pprint
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display
from PIL import Image
from io import BytesIO


def get_coordinates(place_name):
    """
    Fetches latitude and longitude coordinates for a given place using the Google Maps Geocoding API.
    
    Parameters:
    place_name (str): The name of the place for which to fetch coordinates.
    
    Returns:
    tuple: A tuple containing latitude and longitude coordinates, or None if the request failed.
    """
    # Base URL for Google Maps Geocoding API
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    
    # Parameters for API request
    params = {
        'address': place_name,
        'key': gkey  # Replace with your Google API key if replicating
    }
    
    # Send GET request to Google Maps Geocoding API
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:  # Make sure request was successful
        data = response.json()  
        if data['results']:
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            print("No results found for the place name:", place_name)
            return None
    else:
        print("Failed to fetch coordinates for the place:", place_name)
        return None
    
    
def weather_forecast(lat, lon):  
    """
    Fetches weather forecast data for a given location using the OpenWeatherMap API.
    
    Parameters:
    lat (float): Latitude coordinate of the location.
    lon (float): Longitude coordinate of the location.
        
    Returns:
    pandas.DataFrame or None: DataFrame containing daily weather forecast data, or None if the request failed.
    """
    # Base URL for OpenWeatherMap One Call API
    base_url = "https://api.openweathermap.org/data/3.0/onecall"
    
    # API key for accessing OpenWeatherMap data
    api_key = weather_key
    
    # Parameters for API request
    params = {
        'lat': lat,
        'lon': lon,
        'appid': api_key,
        'units': 'metric'  # Use 'imperial' for Fahrenheit
    }
    
    # Send GET request to OpenWeatherApp API
    response = requests.get(base_url, params=params)
    
    if response.status_code != 200: # Make sure request was successful
        print("Failed to fetch forecast data:", response.status_code, response.text)
        return

    data = response.json()

    daily_forecasts = data.get('daily')
    
    # Check if daily forecasts were returned
    if not daily_forecasts:
        print("No daily forecast data found")
        return

    daily_data = []

    for forecast in daily_forecasts: # Extract relevant information from forecast
        daily_data.append([datetime.fromtimestamp(forecast.get('dt')).strftime('%Y-%m-%d'), # Timestamp for the forecast date
                        forecast.get('temp', {}).get('day'),  # Day temperature
                        forecast.get('temp', {}).get('night'),  # Night temperature
                        forecast.get('weather', [{}])[0].get('icon')
                        ])  # Weather description
    
    # Create DataFrame to store processed data
    daily_df = pd.DataFrame(data = daily_data, columns = ['Date', 'Day Temp', 'Night Temp', 'Icon'])
    
    # Add 'Day' column with day names
    daily_df.insert(1, 'Day', pd.to_datetime(daily_df['Date']).dt.day_name())
    
    # Drop 'Date' column
    daily_df.drop(columns = ['Date'], inplace=True)

    return daily_df


def plot_temp_variation(data):
    '''
    Plots the temperature of day and night for our data
    
    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing temperature data with columns
                            "Day", "Day Temp", "Night Temp" & "Icon".
    '''
    # Drop the "Icon" column and the first row
    temp = data.drop(index=0, columns=['Icon'])
    temp = temp.reset_index(drop=True)
    
    # Melt the DataFrame to have one column for temperature type (day or night)
    temp_melted = temp.melt(id_vars='Day', var_name='time', value_name='temp')

    # Plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    # Plotting the line graphs for day and night temps
    sns.lineplot(data=temp_melted, x='Day', y='temp', hue='time', 
                 palette={'Day Temp': 'red', 'Night Temp': 'blue'},
                            marker='o', markersize=5)

    # Adding text labels for each data point
    for i, row in temp.iterrows():
        for col in ['Day Temp', 'Night Temp']:
            plt.text(i, row[col], row[col], ha='center', fontsize=12)

    # Setting labels and title
    plt.xlabel('')
    plt.ylabel('Temperature (°C)')

    # Customizing x-axis ticks (abbreviating days)
    abbreviated_days = [day[:3] for day in temp['Day']]
    plt.xticks(range(len(abbreviated_days)), abbreviated_days, ha='right')

    # Remove gridlines
    plt.grid(False)

    # Showing the plot
    plt.tight_layout()
    plt.show()

    
def forecast_icons(forecast):
    '''
    Displays weather icon for given forecast data.
    
    Parameters:
    forecast (pd.DataFrame): A pandas DataFrame containing temperature data with columns
                                "Day", "Day Temp", "Night Temp" & "Icon".
    '''
    # Drop first row
    forecast = forecast.drop(index=0)
    
    # Create subplots for each weather icon
    fig, axs = plt.subplots(1, len(forecast['Icon']), figsize=(10, 2))

    # Loop each icon code in DataFrame
    for i, icon_code in enumerate(forecast['Icon']):
        url = f'https://openweathermap.org/img/wn/{icon_code}@2x.png'
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        axs[i].imshow(img)
        axs[i].axis('off')
    
    # Show the plot with all icons
    plt.show()
    

def get_access_token():
    """
    Obtains an access token from the Amadeus API using client credentials.

    This function sends a POST request to the Amadeus API token endpoint with the client
    credentials to obtain an access token. If the request is successful, the access token
    is printed and returned. If the request fails, an error message is printed.

    Returns:
    str: The access token if the request is successful, otherwise None.

    Note:
    The variables `amadeus_key` and `amadeus_secret` must be defined with the appropriate
    API key and secret before calling this function.
    """
    token_url = 'https://test.api.amadeus.com/v1/security/oauth2/token'
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': amadeus_key,  # Use imported API key
        'client_secret': amadeus_secret  # Use imported API secret
    }
    
    # Send POST request to obtain access token
    response = requests.post(token_url, data=token_data)
    
    if response.status_code == 200:
        # Extract the access token from the response JSON
        access_token = response.json()['access_token']
        print("Access Token:", access_token)
    else:
        # Print error message if the request was not successful
        print("Failed to obtain access token:", response.status_code, response.text)
    return access_token


def flightoffers(origin_code, destination_code, departureDate):
    """
    Fetches flight offers from the Amadeus API and returns a DataFrame with flight details.

    Parameters:
    origin_code (str): The IATA code for the origin airport/city.
    destination_code (str): The IATA code for the destination airport.
    departureDate (str): The departure date in the format 'YYYY-MM-DD'.

    Returns:
    pd.DataFrame: A DataFrame containing flight details including airline, departure airport,
                  departure time, arrival time, arrival airport, price per person, and total price.

    This function performs the following steps:
    1. Sends a GET request to the Amadeus API to fetch flight offers based on the given parameters.
    2. Extracts relevant flight details from the response JSON.
    3. Formats the departure and arrival times.
    4. Creates a DataFrame with the extracted flight details.
    5. Removes duplicate entries and limits the DataFrame to the top 3 flight offers.
    """
    url = 'https://test.api.amadeus.com/v2/shopping/flight-offers'

    params = {
        'originLocationCode': origin_code,
        'destinationLocationCode': destination_code,
        'departureDate': departureDate,
        'maxPrice': 300,
        'adults': 1,
        'max': 50
    }
    
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    # Send GET request to fetch flight offers
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        data = response.json()
    else:
        print("Failed to fetch flight offers:", response.status_code, response.text)

    flight_offers = data.get('data', [])

    flights = []
    for offer in flight_offers:
        # Get details for each flight offer
        itinerary = offer.get('itineraries', [])[0]
        departure_segment = itinerary.get('segments', [])[0]
        arrival_segment = itinerary.get('segments', [])[-1]  # Get the last segment
        price = offer.get('price', {}).get('total')

        price_per_person = float(price)
        
        # Extracting airline details
        airline = departure_segment.get('carrierCode')

        # Extracting departure details
        departure = departure_segment.get('departure')
        departure_airport = departure.get('iataCode')
        departure_time = departure.get('at')

        # Extracting arrival details
        arrival = arrival_segment.get('arrival')
        arrival_airport = arrival.get('iataCode')
        arrival_time = arrival.get('at')

        # Formatting departure and arrival times
        departure_time_formatted = datetime.strptime(departure_time, "%Y-%m-%dT%H:%M:%S").strftime("%B %dth %H:%M:%S")
        arrival_time_formatted = datetime.strptime(arrival_time, "%Y-%m-%dT%H:%M:%S").strftime("%B %dth %H:%M:%S")

        # Append flight details to the list
        flights.append([airline,
                        departure_airport,
                        departure_time_formatted,
                        arrival_time_formatted,
                        arrival_airport,
                        price_per_person,
                        price_per_person * 2
                       ])
        
    # Define column names for DataFrame
    df_columns = ['Airline', 'Dep Airport', 'Dep Time', 'Arr Time', 'Arr Airport', 'Price PP', 'Total Price']
    flights_df = pd.DataFrame(data = flights, columns = df_columns)
    
    # Remove duplicates
    flights_df = flights_df.drop_duplicates().reset_index(drop=True)
    
    # Filter to top 3 flight offers
    flights_df = flights_df.head(3)
    
    return flights_df


def get_iata_code(location_name):
    """
    Retrieves the IATA code for a given location name using the Amadeus API.
    
    Parameters:
    location_name (str): The name of the location (e.g., city or airport name).
    
    Returns:
    str: The IATA code of the location.
    """
    url = 'https://test.api.amadeus.com/v1/reference-data/locations'
    params = {
        'keyword': location_name,
        'subType': 'AIRPORT,CITY',
    }
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if data['data']:
            iata_code = data['data'][0]['iataCode']
            return iata_code
        else:
            print("No IATA code found for the given location.")
            return None
    else:
        print("Failed to fetch IATA code:", response.status_code, response.text)
        return None
    
    
def get_info(city, keyw):
    '''
    Get a DataFrame with information about the top 5 keyw results from city.
    
    Inputs:
        city (str): The name of the city for which to search for.
        keyw (str): The keyword to search for within the city (e.g Restaurant, Hotel)
        
    Outputs: 
        DataFrame: DataFrame with name, address, rating, N user ratings, latitude, and longitude.
    '''
    # Get coordinates for the city
    coordinates = get_coordinates(city)
    if coordinates is None:
        return None
    
    query = f'Best {keyw} in {city}'
    params = {
        "query": query,
        "location": f'{coordinates[0]},{coordinates[1]}',
        "key": gkey  # Replace with your Google API key if replicating
    }

    # Build URL using the Google Maps API
    base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    
    # Send GET request to Google Places API
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        results = response.json().get('results', [])[:10]

        places_list = []
        for place in results:
            truncated_name = place['name'][:30] + '...' if len(place['name']) > 30 else place['name']
            places_list.append([  # Append place details to list
                truncated_name, 
                place.get('formatted_address', 'N/A'),
                place.get('rating'),
                place.get('user_ratings_total'),
                place['geometry']['location']['lat'],
                place['geometry']['location']['lng']
            ])
            
        # Create DataFrame with list of places
        df = pd.DataFrame(data=places_list, columns=['Name', 'Address', 'Rating', 'N User Ratings', 'Lat', 'Lng'])
        return df
    else:
        print("Failed to fetch data from the Google Places API:", response.status_code)
        return None


def add_markers(map_, df, category):
    '''
    Add markers to a Folium map for each location in the DataFrame
    
    Parameters:
    map_ (folium.Map): The Folium map to which markers will be added.
    df (pd.DataFrame): DataFrame containing location data with columns 'Name', 'Address', 'Rating', 'Lat', and 'Lng'.
    category (str): The category of the locations (e.g., 'Bar', 'Restaurant', 'Hotel', 'Cafe', 'Nightclub') 
    to customize marker icons.
    '''
    for idx, row in df.iterrows():
        popup_text = f"Name:{row['Name']}<br>Address: {row['Address']}<br>Rating: {row['Rating']}"

        # Customize marker icons based on category using Leaflet icon names
        if category == 'Bar':
            icon = folium.Icon(color='orange', icon='glyphicon glyphicon-glass')
        elif category == 'Restaurant':
            icon = folium.Icon(color='red', icon='glyphicon glyphicon-cutlery')
        elif category == 'Hotel':
            icon = folium.Icon(color='blue', icon='bed', prefix='fa')
        elif category == 'Cafe':
            icon = folium.Icon(color='green', icon='mug-saucer', prefix='fa')
        else:
            icon = folium.Icon(color='purple', icon='music', prefix='fa')  # Default icon

        folium.Marker(
            location=[row['Lat'], row['Lng']],
            popup=popup_text,
            icon=icon
        ).add_to(map_)


def marker_place(place_name):
    """
    Fetches information about cafes, bars, restaurants, hotels, and nightclubs in a given place.

    Parameters:
    place_name (str): The name of the place for which to fetch information.

    Returns:
    list: A list of DataFrames, each containing information about a specific category of places.
    """
    cafes = get_info(place_name, 'Cafes')
    bars = get_info(place_name, 'Bars')
    res = get_info(place_name, 'Restaurants')
    hotels = get_info(place_name, 'Hotels')
    nclubs = get_info(place_name, 'Nightclubs')
    
    return [cafes, bars, res, hotels, nclubs]


def map_with_markers(place_name):
    """
    Creates a Folium map centered around the average latitude and longitude 
    of all locations in the given DataFrames, and adds markers for each 
    DataFrame with specified colors.
    
    Parameters:
    place_name (str): The name of the place for which to create the map.
        
    Returns:
    folium.Map: Folium map object with markers added.
    """
    df_list = marker_place(place_name)
    
    # Calculate the average latitude and longitude of all locations
    avg_lat = sum(df['Lat'].mean() for df in df_list) / len(df_list)
    avg_lng = sum(df['Lng'].mean() for df in df_list) / len(df_list)
    
    # Initialize a Folium map centered around the average latitude and longitude
    map_ = folium.Map(location=[avg_lat, avg_lng], zoom_start=9)
    
    # Add markers for restaurants with coffee icon
    add_markers(map_, df_list[0], 'Cafe')
    
    # Add markers for bars with wine glass icon
    add_markers(map_, df_list[1], 'Bar')

    # Add markers for restaurants with fork & knife icon
    add_markers(map_, df_list[2], 'Restaurant')
    
    # Add markers for restaurants with Hotel icon
    add_markers(map_, df_list[3], 'Hotel')
    
    # Add markers for restaurants with default icon
    add_markers(map_, df_list[4], 'Nightclubs')

    return map_



# WeiMal Escapes

# Set up Streamlit page title and favicon
st.set_page_config(page_title="WeiMal Escapes", page_icon="✈️")

# Title and description
st.title("Travel App")
st.write("Welcome to the Travel App! Use this app to explore flights, weather, and places of interest.")

# Sidebar
st.sidebar.header("User Input")

# Input fields in the sidebar
place_name = st.sidebar.text_input("Enter a location:", "New York")
departure_date = st.sidebar.date_input("Departure Date")

# Button to trigger the search
search_button = st.sidebar.button("Search")

# Main content area
if search_button:
    st.header("Flight Offers")
    
    # Get access token
    access_token = get_access_token()
    
    # Get IATA code for the place
    origin_code = get_iata_code(place_name)
    
    # Fetch flight offers
    if origin_code and departure_date:
        st.write(f"Showing flight offers from {place_name} on {departure_date}")
        flight_df = flightoffers(origin_code, "LAX", str(departure_date))
        st.dataframe(flight_df)
    else:
        st.warning("Please enter a valid location and departure date.")
    
    # Weather forecast
    st.header("Weather Forecast")
    
    # Get coordinates for the place
    lat, lon = get_coordinates(place_name)
    
    # Fetch weather forecast
    if lat and lon:
        weather_df = weather_forecast(lat, lon)
        st.dataframe(weather_df)
        
        # Plot temperature variation
        st.subheader("Temperature Variation")
        plot_temp_variation(weather_df)
        
        # Display forecast icons
        st.subheader("Forecast Icons")
        forecast_icons(weather_df)
    else:
        st.warning("Weather forecast data not available for the selected location.")
    
    # Places of interest
    st.header("Places of Interest")
    
    # Display map with markers
    st.subheader("Map with Markers")
    map_ = map_with_markers(place_name)
    folium_static(map_)

