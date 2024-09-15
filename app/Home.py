import streamlit as st

st.set_page_config(
    page_title="Real Estate Project",
    page_icon="👋",
)


# Set the main title of the page
st.title("🏠 Welcome to the Real Estate Insight & Prediction Platform! 🏠")

# Overview section with styled text
st.markdown("""
### Overview: 
Our platform offers a comprehensive solution for real estate enthusiasts, buyers, and sellers. 
Using cutting-edge **data scraping** and **machine learning techniques**, we bring you:
- Accurate **price predictions**
- Insightful **market analysis**
- Personalized **recommendations**

All of this to help you make **informed real estate decisions**.
""")

# Key Features Section
st.subheader("✨ Key Features:")

# Price Prediction Model section
st.markdown("""
#### 1. Price Prediction Model:
Get accurate price estimates for flats and houses based on essential features like:
- Number of bedrooms and bathrooms
- Age of the property
- Furnishing status
- Possession details
- And more...

Simply enter the details of the property, and our model will predict its market value based on historical data and trends.
""")

# Data Analysis & Insights section
st.markdown("""
#### 2. Data Analysis & Insights:
Explore detailed analysis of real estate trends. We provide:
- Visual insights into the market data
- Price distribution across different sectors
- Trends in property age, possession, and furnishing type
- Comparison between different property types (e.g., flats, houses)

Our **interactive charts** and **plots** help you understand the market dynamics at a glance.
""")

# Society Recommendation Module section
st.markdown("""
#### 3. Society Recommendation Module:
Looking for similar societies? Based on your input society, our recommendation engine suggests other societies with similar characteristics.

This helps users discover properties in **similar locations** or communities that meet their criteria.
""")

# Why Use Our Platform Section
st.subheader("💡 Why Use Our Platform?")

st.markdown("""
- **Accurate Predictions:** We leverage **data-driven machine learning models** for reliable property valuations.
- **Actionable Insights:** Our analysis tools empower you to make **well-informed decisions**.
- **Personalized Recommendations:** Find the perfect society with our **custom recommendations**.
""")

# Call to action
st.markdown("### 🚀 Get Started Now! 🚀")

