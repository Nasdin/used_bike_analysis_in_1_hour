import asyncio
import calendar
import os
import re
import sqlite3
import tempfile
import time
from datetime import datetime
from urllib.parse import urljoin

import aiohttp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from bs4 import BeautifulSoup

st.set_page_config(page_title="Used Motorbike Price Analyser", page_icon="🏍️", menu_items={
    "About": "This is a simple app to analyze used motorbike prices from SGBikeMart. "
             "It was built within an hour, and then fixed over 3 hours."
             "By Din. https://github.com/Nasdin/used_bike_analysis_in_1_hour",
    "Report a Bug": "https://github.com/Nasdin/used_bike_analysis_in_1_hour"})


class BikeURLGenerator:
    @staticmethod
    def generate(bike_model="", bike_type="", price_from="", price_to="", license_class="",
                 reg_year_from="1970", reg_year_to="2024", monthly_from="", monthly_to="",
                 user="", status=10, category="", page=1):
        base_url = "https://sgbikemart.com.sg/listing/usedbikes/listing/"
        query_params = (
            f"?page={page}&bike_model={bike_model}&bike_type={bike_type}&price_from={price_from}"
            f"&price_to={price_to}&license_class={license_class}&reg_year_from={reg_year_from}"
            f"&reg_year_to={reg_year_to}&monthly_from={monthly_from}&monthly_to={monthly_to}"
            f"&user={user}&status={status}&category={category}"
        )
        return base_url + query_params


class DepreciationStrategy:
    @staticmethod
    def calculate(price, total_months_left):
        if total_months_left == 'N/A' or total_months_left <= 0:
            return {'annual_depreciation': 'N/A', 'monthly_depreciation': 'N/A'}

        if price is np.nan:
            return {'annual_depreciation': 'N/A', 'monthly_depreciation': 'N/A'}

        end_value = price * 0.10
        total_depreciation = price - end_value
        monthly_depreciation = total_depreciation / total_months_left
        annual_depreciation = monthly_depreciation * 12

        return {
            'annual_depreciation': annual_depreciation,
            'monthly_depreciation': monthly_depreciation
        }


def split_currency_value(amount_str):
    # Identify where the numerical part starts
    i = 0
    for i, char in enumerate(amount_str):
        if char.isdigit():
            break

    # Split the string into currency and value
    currency = amount_str[:i].strip()
    currency = re.sub(r'[^A-Za-z]', '', currency)  # Remove any non-letter characters (like $)
    try:
        value = amount_str[i:].replace(",", "").strip()  # Removing commas if any
        value = ''.join(filter(str.isdigit, value))
        value = float(value)
    except:
        value = np.nan

    return {"Currency": currency, "Price": value}


class BikeAnalyzer:
    def __init__(self, depreciation_strategy, projection_strategy):
        self.depreciation_strategy = depreciation_strategy
        self.projection_strategy = projection_strategy
        self.bike_data = {}
        self.lowest_dealer_price_seen = float('inf')

    def update_lowest_dealer_price_seen(self, price):
        if price < self.lowest_dealer_price_seen:
            self.lowest_dealer_price_seen = price

    @staticmethod
    def extract_sgbikemart_id(url):
        return url.split("/")[-2]

    def recommend_low_price(self, bike_data):
        try:
            recommended_low_price = self.lowest_dealer_price_seen / bike_data['Dealer'] * bike_data['Price']
            delta_low_price = recommended_low_price - bike_data['Price']
        except:
            monthly_depreciations = [b["monthly_depreciation"] for b in self.bike_data.values() if
                                     b["monthly_depreciation"] is not np.nan]
            average_monthly_depreciation = np.mean(monthly_depreciations)
            recommended_low_price = (self.lowest_dealer_price_seen + (
                    bike_data['Total Months Left'] * average_monthly_depreciation)) / 2
            delta_low_price = np.nan
        return recommended_low_price, delta_low_price

    def recommend_middle_price(self, bike_data, recommended_low_price):
        try:
            recommended_middle_price = (recommended_low_price + bike_data['Price']) / 2
            delta_middle_price = recommended_middle_price - bike_data['Price']
        except:
            monthly_depreciations = [b["monthly_depreciation"] for b in self.bike_data.values() if
                                     b["monthly_depreciation"] is not np.nan]
            average_monthly_depreciation = np.mean(monthly_depreciations)
            recommended_middle_price = (bike_data['Total Months Left'] * average_monthly_depreciation)
            delta_middle_price = np.nan
        return recommended_middle_price, delta_middle_price

    async def analyze(self, session, url):
        bike_info = await extract_bike_info(session, url)
        bike_depreciation = self.depreciation_strategy.calculate(bike_info["Price"], bike_info['Total Months Left'])
        bike_info.update(bike_depreciation)

        monthly_prices, yearly_prices = self.projection_strategy.project(
            bike_info["Registration Date"], bike_info["COE Expiry Date"],
            bike_info["Price"], bike_depreciation["monthly_depreciation"]
        )

        try:
            bike_info["Dealer"] = list(monthly_prices.values())[0]
            self.update_lowest_dealer_price_seen(bike_info["Dealer"])
            bike_info["monthly_price_data"] = monthly_prices
            bike_info["yearly_price_data"] = yearly_prices
        except AttributeError:
            bike_info["Dealer"] = np.nan
            bike_info["monthly_price_data"] = np.nan
            bike_info["yearly_price_data"] = np.nan

        self.bike_data[self.extract_sgbikemart_id(url)] = bike_info

        return bike_info


async def extract_bike_info(session, bike_url):
    async with session.get(bike_url) as response:
        html_content = await response.text()
    soup = BeautifulSoup(html_content, 'html.parser')

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
        "URL": bike_url
    }
    bike_info.update(split_currency_value(price))

    # Returning the dictionary
    return bike_info


class PriceProjectionStrategy:
    @staticmethod
    def project(registration_date, coe_expiry_date, current_price, monthly_depreciation):
        registration_date = datetime.strptime(registration_date, '%d/%m/%Y')
        coe_expiry_date = datetime.strptime(coe_expiry_date.split()[0], '%d/%m/%Y')

        current_date = datetime.today()
        total_months_since_registration = (
                (current_date.year - registration_date.year) * 12 +
                current_date.month - registration_date.month
        )

        try:
            original_price = current_price + (total_months_since_registration * monthly_depreciation)
        except TypeError:
            return [np.nan], [np.nan]

        monthly_prices, yearly_prices = {}, {}
        current_price, current_date = original_price, registration_date

        while current_date <= coe_expiry_date:
            date_str = current_date.strftime('%d/%m/%Y')
            monthly_prices[date_str] = current_price
            if current_date.month == registration_date.month or current_date == coe_expiry_date:
                yearly_prices[date_str] = current_price

            current_price -= monthly_depreciation
            next_month = current_date.month + 1 if current_date.month < 12 else 1
            next_year = current_date.year + 1 if current_date.month == 12 else current_date.year
            max_day_in_month = calendar.monthrange(next_year, next_month)[1]

            day = min(registration_date.day, max_day_in_month)
            current_date = datetime(year=next_year, month=next_month, day=day)

        expiry_date_str = coe_expiry_date.strftime('%d/%m/%Y')
        monthly_prices[expiry_date_str] = current_price
        yearly_prices[expiry_date_str] = current_price

        return monthly_prices, yearly_prices


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


async def get_current_coe_price(session):
    url = "https://sgbikemart.com.sg"
    async with session.get(url) as response:
        html_content = await response.text()
    return split_currency_value(extract_coe_price(html_content))["Price"]


async def extract_bike_listing_urls(session, base_url):
    """
    Extracts and returns a list of absolute URLs for bike listings from the given webpage.

    Parameters:
    base_url (str): The base URL of the page to scrape.

    Returns:
    list: A list of absolute URLs pointing to individual bike listings.
    """
    async with session.get(base_url) as response:
        html_content = await response.text()

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


class DatabaseFactory:
    initial_data = {
        "Honda": ["CB125", " ", "MSX125", "PCX150", "PCX160", "CV200X", "CV190R", "RX-X 150", "CRF150L", "ADV 150",
                  "ADV 160",
                  "ADV 350", "CB400F", "CBR500R"],
        "Yamaha": ["Aerox 155", "", "Aerox 155 R", "FZS150", "Sniper 150", "MT-15", "X1-R 135", "XSR155"],
        "Suzuki": ["Address 110", ""]
    }

    def __init__(self, db_path="./motorbike_data.db"):
        self.conn = sqlite3.connect(db_path)
        self._initialize_tables()

    def _initialize_tables(self):
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS brands (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    name TEXT UNIQUE NOT NULL
                                )''')
            self.conn.execute('''CREATE TABLE IF NOT EXISTS models (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    brand_id INTEGER NOT NULL,
                                    name TEXT NOT NULL,
                                    FOREIGN KEY (brand_id) REFERENCES brands(id)
                                )''')

            # Check if the tables are empty, and prepopulate if necessary
            cursor = self.conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM brands')
            if cursor.fetchone()[0] == 0:
                self.prepopulate_db()

    def prepopulate_db(self):
        for brand, models in self.initial_data.items():
            self.insert_brand(brand, silence=True)
            for model in models:
                self.insert_model(brand, model, silence=True)

    def insert_brand(self, brand_name, silence=False):
        brand_name = brand_name.capitalize()
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM brands WHERE name = ?', (brand_name,))
        if not cursor.fetchone():
            with self.conn:
                self.conn.execute('INSERT INTO brands (name) VALUES (?)', (brand_name,))
            if not silence:
                st.info(f"Brand '{brand_name}' added.")
        else:
            if not silence:
                st.warning(f"Brand '{brand_name}' already exists.")

    def insert_model(self, brand_name, model_name, silence=False):
        brand_name, model_name = brand_name.capitalize(), model_name.capitalize()
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM brands WHERE name = ?', (brand_name,))
        brand_id = cursor.fetchone()

        if brand_id:
            cursor.execute('SELECT id FROM models WHERE brand_id = ? AND name = ?', (brand_id[0], model_name))
            if not cursor.fetchone():
                with self.conn:
                    self.conn.execute('INSERT INTO models (brand_id, name) VALUES (?, ?)', (brand_id[0], model_name))
                if not silence:
                    st.info(f"Model '{model_name}' added under brand '{brand_name}'.")
            else:
                if not silence:
                    st.info(f"Model '{model_name}' already exists under brand '{brand_name}'.")
        else:
            if not silence:
                st.warning(f"Brand '{brand_name}' not found. Please add the brand first.")

    def get_all_brands_sorted(self):
        cursor = self.conn.cursor()
        cursor.execute('SELECT name FROM brands ORDER BY name ASC')
        return [row[0] for row in cursor.fetchall()]

    def get_all_models_sorted(self, brand_name):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT models.name 
            FROM models 
            JOIN brands ON models.brand_id = brands.id 
            WHERE brands.name = ?
            ORDER BY models.name ASC
        ''', (brand_name.capitalize(),))
        return [row[0] for row in cursor.fetchall()]

    def remove_model(self, brand_name, model_name):
        brand_name, model_name = brand_name.capitalize(), model_name.capitalize()
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM brands WHERE name = ?', (brand_name,))
        brand_id = cursor.fetchone()

        if brand_id:
            cursor.execute('SELECT id FROM models WHERE brand_id = ? AND name = ?', (brand_id[0], model_name))
            model_id = cursor.fetchone()
            if model_id:
                with self.conn:
                    self.conn.execute('DELETE FROM models WHERE id = ?', (model_id[0],))
                st.info(f"Model '{model_name}' removed from brand '{brand_name}'.")
            else:
                st.warning(f"Model '{model_name}' does not exist under brand '{brand_name}'.")
        else:
            st.error(f"Brand '{brand_name}' not found.")

    def remove_empty_brand(self, brand_name):
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM brands WHERE name = ?', (brand_name.capitalize(),))
        brand_id = cursor.fetchone()[0]
        cursor.execute('''
                        SELECT id FROM models 
                        WHERE brand_id = ? 
                    ''', (brand_id,))
        model_exists = cursor.fetchone()
        if not model_exists:
            cursor.execute('''
                            DELETE FROM brands WHERE id = ?
                        ''', (brand_id,))
            st.warning(f"Brand '{brand_name}' removed.")


async def save_image_from_url(session, image_url, image_name, base_url="https://sgbikemart.com.sg"):
    try:
        # Ensure the URL is absolute
        if not image_url.startswith(('http://', 'https://')):
            image_url = urljoin(base_url, image_url)

        async with session.get(image_url) as response:

            if response.status == 200:
                # Create a temporary directory
                temp_dir = tempfile.gettempdir()
                image_path = os.path.join(temp_dir, image_name)

                # Save image to the temporary directory
                with open(image_path, 'wb') as f:
                    f.write(await response.read())

                return image_path
            else:
                st.warning(f"Failed to download image: {image_url}")
                return None

    except Exception as e:
        st.error(f"Error saving image: {e}")
        return None


async def display_bike_images(session, image_url, title):
    image_name = image_url.split("/")[-1]  # Extract the image name from the URL
    image_path = await save_image_from_url(session, image_url, image_name)

    if image_path:
        image = Image.open(image_path)
        st.image(image, caption=title, width=500)


def safe_format(value):
    if value is np.nan:
        return "N/A"
    elif isinstance(value, str):
        try:
            value = float(value)
            return f"${value:.2f}"
        except:
            return "N/A"
    else:
        return f"${value:.2f}"


def display_bike_title_and_links(bike_data):
    title = bike_data["Title"]
    st.title(title)

    col1, col2 = st.columns(2)
    listing_id = BikeAnalyzer.extract_sgbikemart_id(bike_data["URL"])
    col1.caption(f"SGBike Mart Listing ID: {listing_id}")
    col2.markdown(f"[SGBike Mart Listing]({bike_data['URL']})")


def display_metrics(bike_data):
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Analytics: Annual Depreciation",
            value=safe_format(bike_data['annual_depreciation']),
            help="Estimated based on the bike becoming 10% of its value at end of COE"
        )
    with col2:
        st.metric(
            label="Analytics: Monthly Depreciation",
            value=safe_format(bike_data['monthly_depreciation']),
            help="The 'True' cost of your bike every month, as it will become close to worthless at end of its COE lifespan"
        )

    col3, col4 = st.columns(2)
    with col3:
        st.metric(
            label="Analytics: Dealer's Assumed Original Value",
            value=safe_format(bike_data['Dealer']),
            help="What the dealer is pricing it at if it's new, you should compare with the actual new price"
        )
    with col4:
        st.metric(
            label="Current Asking Price",
            value=safe_format(bike_data['Price']),
            help="You should deduct the asking price based on the % difference between brand new vs dealer's assumed value."
        )
    st.caption(
        "If the dealer is charging much more than what it costs new, you should deduct the difference from the asking price."
    )
    st.caption("e.g New costs 15k, but dealer assumed value is 18k, = 17% difference, so deduct 17% from asking price.")


def display_recommended_prices():
    st.subheader("Analytics: Recommended prices to offer")
    st.caption("We get this estimate by comparing with other listings")
    col5, col6 = st.columns(2)
    st.caption("This offer value will update as we fetch more data!")
    recommended_low_price_placeholder = col5.empty()
    recommended_price_placeholder = col6.empty()
    return recommended_low_price_placeholder, recommended_price_placeholder


def display_bike_details_table(bike_data):
    st.subheader("Bike information")

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
        "SGBike Mart Listing URL": bike_data["URL"],
    }

    details_df = pd.DataFrame(details_data.items(), columns=["Attribute", "Value"])
    st.table(details_df)


def display_price_data(bike_data):
    if bike_data['Price'] is not np.nan:
        with st.expander("Show Monthly Price Data"):
            monthly_df = pd.DataFrame.from_dict(bike_data["monthly_price_data"], orient='index', columns=["Price"])
            st.dataframe(monthly_df)
        with st.expander("Show Yearly Price Data"):
            yearly_df = pd.DataFrame.from_dict(bike_data["yearly_price_data"], orient='index', columns=["Price"])
            st.dataframe(yearly_df)


async def display_bike_analysis(session, bike_data):
    display_bike_title_and_links(bike_data)
    await display_bike_images(session, bike_data["Image URL"], bike_data["Title"])
    display_metrics(bike_data)
    recommended_low_price_placeholder, recommended_price_placeholder = display_recommended_prices()
    display_bike_details_table(bike_data)
    display_price_data(bike_data)

    return recommended_low_price_placeholder, recommended_price_placeholder


async def display_coe_price_sidebar(session):
    st.sidebar.header("Current Motorbike COE Price")
    coe_price = await get_current_coe_price(session)
    coe_price_per_month = coe_price / 120
    coe_price_per_year = coe_price / 10
    today_date = pd.Timestamp.now().strftime("%d/%m/%Y")

    st.sidebar.subheader(f"Price as of {today_date}")
    st.sidebar.metric(label="COE Price", value=f"${coe_price:.2f}")
    col1, col2 = st.sidebar.columns(2)
    col1.metric(label="Per Month", value=f"${coe_price_per_month:.2f}")
    col2.metric(label="Per Year", value=f"${coe_price_per_year:.2f}")


# Function to select or add brand and model
def select_or_add_brand_and_model(motorbike_factory):
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Select Brand", options=motorbike_factory.get_all_brands_sorted(), index=0)
        new_brand = st.text_input("Or Enter a New Brand").title()
        if new_brand:
            motorbike_factory.insert_brand(new_brand)
            brand = new_brand

    with col2:
        model = st.selectbox("Select Model", options=motorbike_factory.get_all_models_sorted(brand), index=0)
        new_model = st.text_input("Or Enter a New Model").title()
        if new_model:
            motorbike_factory.insert_model(brand, new_model)
            model = new_model

    return brand, model


# Function to get search filters
def get_search_filters():
    col3, col4 = st.columns(2)
    with col3:
        price_from = st.text_input("Price From", "0")
    with col4:
        price_to = st.text_input("Price To", "")

    license_class = st.selectbox("License Class", ["", "2B", "2A", "2"], index=0)

    col5, col6 = st.columns(2)
    with col5:
        reg_year_from = st.text_input("Registration Year From", "1970")
    with col6:
        reg_year_to = st.text_input("Registration Year To", "2024")

    return {
        "price_from": price_from,
        "price_to": price_to,
        "license_class": license_class,
        "reg_year_from": reg_year_from,
        "reg_year_to": reg_year_to
    }


# Function to fetch bike listing URLs
async def fetch_bike_listing_urls(session, brand, model, filters):
    bike_model_cleaned = f"{brand} {model}".replace(" ", "+")
    bike_listing_urls = []
    page = 1
    while True:
        bike_listings_url = BikeURLGenerator.generate(
            bike_model=bike_model_cleaned,
            price_from=filters['price_from'],
            price_to=filters['price_to'],
            license_class=filters['license_class'],
            reg_year_from=filters['reg_year_from'],
            reg_year_to=filters['reg_year_to'],
            status=10,  # Available
            page=page
        )

        new_bike_listing_urls = await extract_bike_listing_urls(session, bike_listings_url)
        if not new_bike_listing_urls:
            break
        bike_listing_urls.extend(new_bike_listing_urls)
        page += 1

    return bike_listing_urls


# Function to handle no results found
async def handle_no_results(session, motorbike_factory, brand, model):
    st.warning("No bikes found with the given criteria.")
    test_url = BikeURLGenerator.generate(bike_model=f"{brand} {model}".replace(" ", "+"),
                                         license_class="")
    if not await extract_bike_listing_urls(session, test_url):
        st.warning(
            f"We could not find any bikes for {brand} {model}. Consider trying a different model or checking for spacing errors."
            "e.g MSX125 instead of MSX 125 and ADV 150 instead of ADV150.\n This bike model will be removed from the database"
        )
        if model:
            motorbike_factory.remove_model(brand, model)
        if brand:
            motorbike_factory.remove_empty_brand(brand)


def update_recommended_placeholders(analyzed_bikes, bike_analyzer):
    """Updates the placeholders for recommended prices for all analyzed bikes."""
    for bike_data, placeholder_low, placeholder_rec in analyzed_bikes:
        recommended_low_price, delta_low_price = bike_analyzer.recommend_low_price(bike_data)
        recommended_middle_price, delta_middle_price = bike_analyzer.recommend_middle_price(bike_data,
                                                                                            recommended_low_price)

        if delta_low_price < 0:
            formatted_delta_low_price = f"-${abs(delta_low_price):.2f}"
        else:
            formatted_delta_low_price = f"${delta_low_price:.2f}"

        # Adjusting delta_middle_price to handle negative values
        if delta_middle_price < 0:
            formatted_delta_middle_price = f"-${abs(delta_middle_price):.2f}"
        else:
            formatted_delta_middle_price = f"${delta_middle_price:.2f}"

        # Updated code with the adjusted deltas
        placeholder_low.metric(
            label="Recommended Low Price",
            value=f"${recommended_low_price:.2f}",
            delta=formatted_delta_low_price,
            delta_color="inverse",
            help=f"Based on the lowest dealer price found of {bike_analyzer.lowest_dealer_price_seen:.2f}"
        )

        placeholder_rec.metric(
            label="Recommended Middle Price",
            value=f"${recommended_middle_price:.2f}",
            delta=formatted_delta_middle_price,
            delta_color="inverse",
            help="Middle price between the lowest dealer price and the current asking price"
        )


# Function to initialize charts
def initialize_charts():
    scatter_data = pd.DataFrame(columns=["COE Months Left", "Price"])
    depreciation_data = pd.DataFrame(columns=["COE Months Left", "Annual Depreciation"])

    st.subheader("Price over COE left in bikes")
    scatter_chart = st.scatter_chart(scatter_data, use_container_width=True,
                                     x="COE Months Left", y="Price")

    st.subheader("Depreciation over COE left in bikes")
    depreciation_chart = st.scatter_chart(depreciation_data, use_container_width=True,
                                          x="COE Months Left", y="Annual Depreciation")

    return scatter_chart, depreciation_chart


async def display_search_results(session, bike_listing_urls, bike_analyzer, brand, model, sidebar_data):
    """Displays the search results and analyzes each bike."""
    st.subheader(f"Found {len(bike_listing_urls)} bikes:")
    st.header("Analyzing Bikes...")
    progress_bar1 = st.progress(0)
    st.caption("Scroll down to see details")

    scatter_chart, depreciation_chart = initialize_charts()
    st.title(f"All {brand} {model} analysis")

    analyzed_bikes = []
    progress_bar = st.progress(0)
    total_bikes = len(bike_listing_urls)

    analysis_tasks = [bike_analyzer.analyze(session, url) for url in bike_listing_urls]
    for i, analysis_task in enumerate(asyncio.as_completed(analysis_tasks)):
        bike_data = await analysis_task

        low_placeholder, rec_placeholder = await display_bike_analysis(session, bike_data)
        sidebar_data = update_sidebar_if_lowest(bike_data, bike_analyzer.lowest_dealer_price_seen, sidebar_data)

        analyzed_bikes.append((bike_data, low_placeholder, rec_placeholder))

        if bike_data["Price"] is not np.nan:
            update_charts(scatter_chart, depreciation_chart, bike_data)

        # Update recommended placeholders
        update_recommended_placeholders(analyzed_bikes, bike_analyzer)
        progress_bar.progress((i + 1) / len(bike_listing_urls), text=f"{i + 1}/{total_bikes} bikes fetched")
        progress_bar1.progress((i + 1) / len(bike_listing_urls), text=f"{i + 1}/{total_bikes} bikes fetched")
        time.sleep(0.05)

    st.success("All bike details fetched and displayed.")
    st.info("You can now select another bike to analyze.")


def update_charts(scatter_chart, depreciation_chart, bd):
    new_scatter_data = pd.DataFrame({
        "COE Months Left": [float(bd['Total Months Left'])],
        "Price": [bd['Price']]
    })
    scatter_chart.add_rows(new_scatter_data.set_index("COE Months Left"))

    new_depreciation_data = pd.DataFrame({
        "COE Months Left": [float(bd['Total Months Left'])],
        "Annual Depreciation": [bd['annual_depreciation']]
    })
    depreciation_chart.add_rows(new_depreciation_data.set_index("COE Months Left"))


def display_analytics_sidebar():
    st.sidebar.header("Analytics Summary")
    st.sidebar.caption("Most price-to-value motorbike found")
    sidebar_model = st.sidebar.empty()
    sidebar_posting_id = st.sidebar.empty()
    sidebar_price = st.sidebar.empty()
    col1, col2 = st.sidebar.columns(2)
    sidebar_annual_depreciation = col1.empty()
    sidebar_monthly_depreciation = col2.empty()
    sidebar_bike_coe_left = st.sidebar.empty()
    sidebar_link = st.sidebar.empty()

    sidebar_data = {

        "sidebar_model": sidebar_model,
        "sidebar_posting_id": sidebar_posting_id,
        "sidebar_price": sidebar_price,
        "sidebar_monthly_depreciation": sidebar_monthly_depreciation,
        "sidebar_annual_depreciation": sidebar_annual_depreciation,
        "sidebar_bike_coe_left": sidebar_bike_coe_left,
        "sidebar_link": sidebar_link

    }

    return sidebar_data


def update_sidebar_if_lowest(bd, lowest_dealer_price, sidebar_data):
    current_dealer_price = bd['Dealer']
    if current_dealer_price <= lowest_dealer_price:
        sidebar_data['sidebar_model'].subheader(bd['Title'])
        sidebar_data['sidebar_price'].metric(label="Asking Price", value=f"${bd['Price']:.2f}")
        sidebar_data['sidebar_monthly_depreciation'].metric(label="Monthly Depreciation",
                                                            value=f"${bd['monthly_depreciation']:.2f}")
        sidebar_data['sidebar_annual_depreciation'].metric(label="Annual Depreciation",
                                                           value=f"${bd['annual_depreciation']:.2f}")
        sidebar_data['sidebar_bike_coe_left'].metric(label="COE Left", value=bd['Years & Months Left'])
        sidebar_data['sidebar_link'].markdown(f"[SGBike Mart Listing]({bd['URL']})")
        sidebar_data['sidebar_posting_id'].caption(f"Listing ID: {BikeAnalyzer.extract_sgbikemart_id(bd['URL'])}")

    return sidebar_data


async def main():
    # Streamlit app title and subtitle
    st.title("Used Motorbikes Scanner")
    st.caption("For TechOverflow (A 1 hour hackathon by Din)")

    # Initialize necessary objects
    motorbike_factory = DatabaseFactory()
    bike_analyzer = BikeAnalyzer(DepreciationStrategy(), PriceProjectionStrategy())
    async with aiohttp.ClientSession() as session:
        # Sidebar: Display COE Price
        await display_coe_price_sidebar(session)

        # Sidebar: Analytics Summary
        sidebar_data = display_analytics_sidebar()  # Initialize sidebar placeholders

        # User Input: Select or Add Brand and Model
        brand, model = select_or_add_brand_and_model(motorbike_factory)

        # Additional search filters with defaults
        filters = get_search_filters()

        # Search Button
        if st.button("Search"):
            # Fetch bike listing URLs based on brand, model, and filters
            bike_listing_urls = await fetch_bike_listing_urls(session, brand, model, filters)

            if not bike_listing_urls:
                # Handle the case where no results are found
                await handle_no_results(session, motorbike_factory, brand, model)
            else:
                # Display search results and analyze the bikes
                await display_search_results(session, bike_listing_urls, bike_analyzer, brand, model, sidebar_data)


if __name__ == "__main__":
    asyncio.run(main())
