import calendar
import os
import re
import tempfile
import time
from datetime import datetime
from urllib.parse import urljoin

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup

st.set_page_config(page_title="Used Motorbike Price Analyser", page_icon="🏍️")


def generate_used_bike_search_url(
        bike_model="Honda",
        bike_type="",
        price_from="",
        price_to="",
        license_class="2B",
        reg_year_from="1970",
        reg_year_to="2024",
        monthly_from="",
        monthly_to="",
        user="",
        status="50",
        category=""
):
    base_url = "https://sgbikemart.com.sg/listing/usedbikes/listing/"
    query_params = (
        f"?bike_model={bike_model}&bike_type={bike_type}&price_from={price_from}"
        f"&price_to={price_to}&license_class={license_class}&reg_year_from={reg_year_from}"
        f"&reg_year_to={reg_year_to}&monthly_from={monthly_from}&monthly_to={monthly_to}"
        f"&user={user}&status={status}&category={category}"
    )
    return base_url + query_params


def calculate_depreciation(price, total_months_left):
    # If there are no remaining months, return 'N/A'
    if total_months_left == 'N/A' or total_months_left <= 0:
        return {'annual_depreciation': 'N/A', 'monthly_depreciation': 'N/A'}

    # Calculate depreciated value at the end of COE (10% of initial price)
    end_value = price * 0.10

    # Calculate total depreciation amount (price reduction needed)
    total_depreciation = price - end_value

    # Calculate monthly depreciation
    monthly_depreciation = total_depreciation / total_months_left

    # Calculate annual depreciation
    annual_depreciation = monthly_depreciation * 12

    return {
        'annual_depreciation': annual_depreciation,
        'monthly_depreciation': monthly_depreciation
    }


def split_currency_value(amount_str):
    # Identify where the numerical part starts
    for i, char in enumerate(amount_str):
        if char.isdigit():
            break

    # Split the string into currency and value
    currency = amount_str[:i].strip()
    currency = re.sub(r'[^A-Za-z]', '', currency)  # Remove any non-letter characters (like $)

    value = amount_str[i:].replace(",", "").strip()  # Removing commas if any

    return {"Currency": currency, "Price": float(value)}


def extract_bike_info(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extracting information
    title = soup.find('h2', class_='card-title').text.strip()
    price = soup.find('h2', class_='text-center strong').text.strip()

    # Details Table
    details_table = soup.find('table', class_='table mb-0')
    details = {}
    for row in details_table.find_all('tr'):
        key = row.find('td', class_='name').text.strip()
        value = row.find('td', class_='value').text.strip()
        details[key] = value

    # Extracting specific details
    brand = details.get('Brand', 'N/A')
    model = details.get('Model', 'N/A')
    engine_capacity = details.get('Engine Capacity', 'N/A')
    classification = details.get('Classification', 'N/A')
    registration_date = details.get('Registration Date', 'N/A')
    coe_expiry_date = details.get('COE Expiry Date', 'N/A')
    vehicle_type = details.get('Type of Vehicle', 'N/A')

    # Calculating remaining COE
    if coe_expiry_date != 'N/A':
        coe_expiry_date_clean = coe_expiry_date.split()[0]
        coe_expiry_date_obj = datetime.strptime(coe_expiry_date_clean, '%d/%m/%Y')
        today = datetime.today()
        remaining_time = coe_expiry_date_obj - today
        remaining_years = remaining_time.days // 365
        remaining_months = (remaining_time.days % 365) // 30
        total_remaining_months = remaining_time.days / 30
    else:
        remaining_years = remaining_months = total_remaining_months = 'N/A'

    # Extracting description
    description_div = soup.find('div', class_='listing-details')
    description = description_div.text.strip() if description_div else 'N/A'

    # Extracting the bike image URL
    image_div = soup.find('div', class_='slider-item')
    image_style = image_div.get('style') if image_div else ''
    image_url = ''
    if image_style:
        image_url = image_style.split("url('")[1].split("')")[0]

    # Creating the dictionary with the extracted information
    bike_info = {
        "Title": title,
        "Brand": brand,
        "Model": model,
        "Engine Capacity": engine_capacity,
        "Classification": classification,
        "Registration Date": registration_date,
        "COE Expiry Date": coe_expiry_date,
        "Total Months Left": total_remaining_months,
        "Years & Months Left": f"{remaining_years} years, {remaining_months} months",
        "Type of Vehicle": vehicle_type,
        "Description": description,
        "Image URL": image_url,
    }
    bike_info.update(split_currency_value(price))

    # Returning the dictionary
    return bike_info


def project_vehicle_price(registration_date, coe_expiry_date, current_price, monthly_depreciation):
    # Convert registration and COE expiry dates to datetime objects
    registration_date = datetime.strptime(registration_date, '%d/%m/%Y')
    coe_expiry_date = datetime.strptime(coe_expiry_date.split()[0], '%d/%m/%Y')

    # Calculate the number of months the vehicle has already depreciated
    current_date = datetime.today()
    total_months_since_registration = (
                                              current_date.year - registration_date.year) * 12 + current_date.month - registration_date.month

    # Calculate the original price at the registration date
    original_price = current_price + (total_months_since_registration * monthly_depreciation)

    # Create a dictionary to store projected prices for each month
    monthly_prices = {}
    yearly_prices = {}

    # Iterate month by month from the registration date to the COE expiry date (inclusive)
    current_price = original_price
    current_date = registration_date
    while current_date <= coe_expiry_date:
        date_str = current_date.strftime('%d/%m/%Y')
        monthly_prices[date_str] = current_price

        # Add price to yearly_prices only if it's the first month of the year or the COE expiry date
        if current_date.month == registration_date.month or current_date == coe_expiry_date:
            yearly_prices[date_str] = current_price

        # Deduct the monthly depreciation
        current_price -= monthly_depreciation

        # Move to the next month
        next_month = current_date.month + 1 if current_date.month < 12 else 1
        next_year = current_date.year + 1 if current_date.month == 12 else current_date.year
        max_day_in_month = calendar.monthrange(next_year, next_month)[1]

        # Adjust the day if it is out of range
        day = min(registration_date.day, max_day_in_month)
        current_date = datetime(year=next_year, month=next_month, day=day)

    # Ensure the COE expiry date is included in both monthly and yearly prices
    expiry_date_str = coe_expiry_date.strftime('%d/%m/%Y')
    monthly_prices[expiry_date_str] = current_price
    yearly_prices[expiry_date_str] = current_price

    return monthly_prices, yearly_prices


def analyze_used_bike(url):
    bike_info = extract_bike_info(url)
    bike_depreciation = calculate_depreciation(bike_info["Price"], bike_info['Total Months Left'])
    bike_info.update(bike_depreciation)

    monthly_prices, yearly_prices = project_vehicle_price(bike_info["Registration Date"], bike_info["COE Expiry Date"],
                                                          bike_info["Price"], bike_depreciation["monthly_depreciation"])

    # Adds the starting price of the monthly_prices into the bike info as the dealer's assumed bike value
    bike_info["Dealer"] = list(monthly_prices.values())[0]
    bike_info["monthly_price_data"] = monthly_prices
    bike_info["yearly_price_data"] = yearly_prices

    return bike_info


def extract_coe_price(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Locate the card-title with "COE Results"
    coe_card_title = soup.find('div', class_='card-title', string='COE Results')

    if not coe_card_title:
        return None

    # Navigate to the parent div to get the whole COE card content
    coe_card = coe_card_title.find_parent('div', class_='card')

    # Locate the specific strong tag with the price information
    price_tag = coe_card.find('div', class_='col-4').find_all('strong')[1]  # This will be the second strong tag

    if price_tag:
        return price_tag.text.strip()

    return None


def get_current_coe_price():
    url = "https://sgbikemart.com.sg"
    response = requests.get(url)
    return split_currency_value(extract_coe_price(response.content))["Price"]


def extract_bike_listing_urls(base_url):
    """
    Extracts and returns a list of absolute URLs for bike listings from the given webpage.

    Parameters:
    base_url (str): The base URL of the page to scrape.

    Returns:
    list: A list of absolute URLs pointing to individual bike listings.
    """
    response = requests.get(base_url)
    html_content = response.text

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all the <a> tags that lead to the individual bike listings
    bike_links = soup.find_all('a', href=True)
    listing_urls = set()
    # Extract and print full URLs of the bike listings
    for link in bike_links:
        href = link['href']
        if '/listing/usedbike/' in href:
            full_url = urljoin(base_url, href)
            listing_urls.add(full_url)

    return listing_urls


# Global cache for brands and models
@st.cache_data
def get_cached_data():
    return {
        "brands": ["Honda", "Yamaha", "Suzuki"],  # Pre-populate with some brands
        "models": {
            "Honda": ["CB125", "MSX125", "PCX150", "CV200X", "CV190R", "RX-X 150", "CRF150L", "ADV 150", "ADV 350", "CB400F",
                      "CBR500R"],
            "Yamaha": ["Aerox 155", "Aerox 155 R", "FZS150", "Sniper 150", "MT-15", "X1-R 135", "XSR155"],
            "Suzuki": ["Address 110"]}
    }


def save_image_from_url(image_url, image_name, base_url="https://sgbikemart.com.sg"):
    try:
        # Ensure the URL is absolute
        if not image_url.startswith(('http://', 'https://')):
            image_url = urljoin(base_url, image_url)

        response = requests.get(image_url)
        if response.status_code == 200:
            # Create a temporary directory
            temp_dir = tempfile.gettempdir()
            image_path = os.path.join(temp_dir, image_name)

            # Save image to the temporary directory
            with open(image_path, 'wb') as f:
                f.write(response.content)

            return image_path
        else:
            st.warning(f"Failed to download image: {image_url}")
            return None
    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None


def display_bike_images(image_url, title):
    image_name = image_url.split("/")[-1]  # Extract the image name from the URL
    image_path = save_image_from_url(image_url, image_name)

    if image_path:
        image = Image.open(image_path)
        st.image(image, caption=title, use_column_width=True)


def display_bike_analysis(bike_data):
    # Title
    st.title(bike_data["Title"])
    display_bike_images(bike_data["Image URL"], bike_data["Title"])

    # Highlighted metrics: Annual and Monthly Depreciation
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Annual Depreciation", value=f"${bike_data['annual_depreciation']:.2f}")
    with col2:
        st.metric(label="Monthly Depreciation", value=f"${bike_data['monthly_depreciation']:.2f}")

    col3, col4 = st.columns(2)
    # Dealer's Assumed Bike Original Value
    col3.metric(label=f"Dealer's Assumed Bike Original Value",
                value=f"${bike_data['Dealer']:.2f}")
    col3.caption("Assumed bike if new price, compare with what you know it costs if you buy new")
    col3.caption("If this difference is too high, it means the used price might be overpriced.")
    col4.metric(label="Current Asking Price:", value=f"${bike_data['Price']:.2f}")

    col4.caption(
        "You should deduct the asking price based on the % difference between brand new vs dealer's assumed value.")
    col4.caption(
        "e.g New costs 15k, but dealer assumed value is 18k,= 17% difference, so deduct 17% from asking price.")

    # Bike Details in a Table
    details_data = {
        "Brand": bike_data["Brand"],
        "Model": bike_data["Model"],
        "Engine Capacity": bike_data["Engine Capacity"],
        "Classification": bike_data["Classification"],
        "Registration Date": bike_data["Registration Date"],
        "COE Expiry Date": bike_data["COE Expiry Date"],
        "Years & Months Left": bike_data["Years & Months Left"],
        "Type of Vehicle": bike_data["Type of Vehicle"],
        "Price": f"${bike_data['Price']}",
        "Currency": bike_data["Currency"],
        "Description": bike_data["Description"],
    }

    details_df = pd.DataFrame(details_data.items(), columns=["Attribute", "Value"])
    st.table(details_df)

    # Expandable DataFrames for Monthly and Yearly Price Data
    with st.expander("Show Monthly Price Data"):
        monthly_df = pd.DataFrame.from_dict(bike_data["monthly_price_data"], orient='index', columns=["Price"])
        st.dataframe(monthly_df)

        # Plotting the time series
        st.subheader("Monthly Price Over Time")
        plt.figure(figsize=(10, 4))
        plt.plot(monthly_df.index, monthly_df["Price"], marker='o')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Monthly Price Over Time")
        plt.xticks(rotation=45)
        st.pyplot(plt)

    with st.expander("Show Yearly Price Data"):
        yearly_df = pd.DataFrame.from_dict(bike_data["yearly_price_data"], orient='index', columns=["Price"])
        st.dataframe(yearly_df)

        # Plotting the time series
        st.subheader("Yearly Price Over Time")
        plt.figure(figsize=(10, 4))
        plt.plot(yearly_df.index, yearly_df["Price"], marker='o', color='orange')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title("Yearly Price Over Time")
        plt.xticks(rotation=45)
        st.pyplot(plt)


cached_data = get_cached_data()

# Streamlit app title
st.title("Used Motorbike Price Analyser")
st.caption("For TechOverflow (A 1 hour hackathon by Din)")

# Sidebar: Display COE Price
st.sidebar.header("Current Motorbike COE Price")
coe_price = get_current_coe_price()
coe_price_per_month = coe_price / 120
coe_price_per_year = coe_price / 10
today_date = pd.Timestamp.now().strftime("%d/%m/%Y")

st.sidebar.subheader(f"Price as of {today_date}")
st.sidebar.metric(label="COE Price", value=f"${coe_price:.2f}")
st.sidebar.metric(label="Per Month", value=f"${coe_price_per_month:.2f}")
st.sidebar.metric(label="Per Year", value=f"${coe_price_per_year:.2f}")

# User Input: Select or Add Brand
st.subheader("Search for a Used Motorbike")
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Select Brand", options=cached_data["brands"], index=0)
    new_brand = st.text_input("Or Enter a New Brand").title()
    if new_brand:
        if new_brand not in cached_data["brands"]:
            cached_data["brands"].append(new_brand)
            cached_data["models"][new_brand] = []

with col2:
    model = st.selectbox("Select Model", options=cached_data["models"].get(brand, []), index=0)
    new_model = st.text_input("Or Enter a New Model").title()
    if new_model:
        if new_model not in cached_data["models"].get(brand, []):
            cached_data["models"][brand].append(new_model)

# Additional search filters with defaults
col3, col4 = st.columns(2)

with col3:
    price_from = st.text_input("Price From", "0")

with col4:
    price_to = st.text_input("Price To", "")

license_class = st.selectbox("License Class", ["", "2B", "2A", "2"], index=1)

col5, col6 = st.columns(2)

with col5:
    reg_year_from = st.text_input("Registration Year From", "1970")

with col6:
    reg_year_to = st.text_input("Registration Year To", "2024")

# Search Button
if st.button("Search"):

    bike_model_cleaned = f"{brand} {model}".replace(" ", "+")

    bike_listings_url = generate_used_bike_search_url(
        bike_model=bike_model_cleaned,
        price_from=price_from,
        price_to=price_to,
        license_class=license_class,
        reg_year_from=reg_year_from,
        reg_year_to=reg_year_to,
        status=10,  # Available
    )

    bike_listing_urls = extract_bike_listing_urls(bike_listings_url)

    st.subheader(f"Found {len(bike_listing_urls)} bikes:")
    progress_bar = st.progress(0)
    bike_data_list = []

    for i, url in enumerate(bike_listing_urls):
        bd = analyze_used_bike(url)
        bike_data_list.append(bd)

        display_bike_analysis(bd)
        progress_bar.progress((i + 1) / len(bike_listing_urls))
        time.sleep(0.1)  # Just for demonstration of real-time fetching

    st.success("All bike details fetched and displayed.")
    st.info("You can now select another bike to analyze.")
    st.info(f"Scraped the following links {bike_listings_url}")
    st.table(bike_listing_urls)
