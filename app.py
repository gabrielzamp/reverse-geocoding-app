import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
from typing import Dict, Optional
import io

# Set page config
st.set_page_config(
    page_title="Reverse Geocoding Tool",
    page_icon="🌍",
    layout="wide"
)

class ReverseGeocoder:
    def __init__(self, user_agent="streamlit_reverse_geocoder"):
        self.geolocator = Nominatim(user_agent=user_agent)

    def reverse_geocode_coordinate(self, latitude: float, longitude: float) -> Dict[str, str]:
        """
        Convert lat/long coordinates to address components

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate

        Returns:
            Dictionary with address components
        """
        try:
            # Perform reverse geocoding
            location = self.geolocator.reverse(
                f"{latitude}, {longitude}",
                timeout=10,
                language='en'
            )

            if location and location.raw:
                address_data = location.raw.get('address', {})

                # Extract address components
                result = {
                    'full_address': location.address if location.address else '',
                    'house_number': address_data.get('house_number', ''),
                    'street': address_data.get('road', ''),
                    'city': (address_data.get('city') or
                            address_data.get('town') or
                            address_data.get('village') or
                            address_data.get('municipality', '')),
                    'county': address_data.get('county', ''),
                    'state': (address_data.get('state') or
                             address_data.get('province', '')),
                    'postal_code': address_data.get('postcode', ''),
                    'country': address_data.get('country', ''),
                    'country_code': address_data.get('country_code', '').upper()
                }

                # Create formatted street address
                street_parts = []
                if result['house_number']:
                    street_parts.append(result['house_number'])
                if result['street']:
                    street_parts.append(result['street'])
                result['street_address'] = ' '.join(street_parts)

                return result
            else:
                # Return empty structure if no result found
                return self._empty_address_result()

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            # Don't show too many error messages in Streamlit to avoid spam
            return self._empty_address_result()
        except Exception as e:
            # Don't show too many error messages in Streamlit to avoid spam
            return self._empty_address_result()

    def _empty_address_result(self) -> Dict[str, str]:
        """Return empty address structure for failed lookups"""
        return {
            'full_address': '',
            'house_number': '',
            'street': '',
            'street_address': '',
            'city': '',
            'county': '',
            'state': '',
            'postal_code': '',
            'country': '',
            'country_code': ''
        }

    def process_coordinates_dataframe(self, df: pd.DataFrame, progress_callback=None, rate_limit=1.1) -> pd.DataFrame:
        """
        Process DataFrame with coordinates and add address information

        Args:
            df: DataFrame with coordinates
            progress_callback: Function to update progress

        Returns:
            DataFrame with added address columns
        """
        # Check for required columns (case-insensitive)
        df_columns_lower = [col.lower() for col in df.columns]
        lat_col = None
        lon_col = None
        
        # Find latitude column
        for col in df.columns:
            if col.lower() in ['latitude', 'lat', 'y']:
                lat_col = col
                break
        
        # Find longitude column
        for col in df.columns:
            if col.lower() in ['longitude', 'lon', 'lng', 'long', 'x']:
                lon_col = col
                break
        
        if not lat_col or not lon_col:
            raise ValueError(f"Could not find latitude and longitude columns. Found columns: {list(df.columns)}")

        # Initialize new address columns
        address_columns = [
            'full_address', 'house_number', 'street', 'street_address',
            'city', 'county', 'state', 'postal_code', 'country', 'country_code'
        ]

        for col in address_columns:
            df[col] = ''

        # Process each row
        successful_geocodes = 0
        total_rows = len(df)
        error_count = 0
        
        for index, row in df.iterrows():
            latitude = row[lat_col]
            longitude = row[lon_col]

            # Skip if coordinates are missing or invalid
            if pd.isna(latitude) or pd.isna(longitude):
                continue

            # Update progress
            if progress_callback:
                progress_callback(index + 1, total_rows, latitude, longitude)

            # Perform reverse geocoding
            address_info = self.reverse_geocode_coordinate(latitude, longitude)

            # Update DataFrame with address information
            for col in address_columns:
                df.at[index, col] = address_info[col]

            if address_info['full_address']:
                successful_geocodes += 1
            else:
                error_count += 1

            # Rate limiting - be nice to the free service
            time.sleep(rate_limit)

        return df, successful_geocodes


def main():
    # Initialize session state
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    # App header
    st.title("🌍 Reverse Geocoding Tool")
    st.markdown("Convert coordinates (latitude/longitude) to full addresses")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This tool converts latitude/longitude coordinates to full addresses using reverse geocoding.
        
        **Required CSV columns:**
        - Latitude (or lat, y)
        - Longitude (or lon, lng, long, x)
        
        **Added columns:**
        - full_address
        - street_address
        - city, state, postal_code
        - country, country_code
        - And more!
        """)
        
        st.header("⚙️ Settings")
        rate_limit = st.slider("Rate limit (seconds between requests)", 1.0, 3.0, 1.1, 0.1)
        st.info("Higher values reduce API load but increase processing time")

    # File upload
    st.header("📁 Upload Your CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file with coordinates",
        type=['csv'],
        help="Your CSV should contain latitude and longitude columns"
    )

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! Found {len(df)} rows and {len(df.columns)} columns.")
            
            # Display file preview
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Show column information
            st.write("**Columns found:**", ", ".join(df.columns))
            
            # Show coordinate column detection
            df_columns_lower = [col.lower() for col in df.columns]
            lat_col = None
            lon_col = None
            
            # Find latitude column
            for col in df.columns:
                if col.lower() in ['latitude', 'lat', 'y']:
                    lat_col = col
                    break
            
            # Find longitude column
            for col in df.columns:
                if col.lower() in ['longitude', 'lon', 'lng', 'long', 'x']:
                    lon_col = col
                    break
            
            if lat_col and lon_col:
                st.success(f"✅ Coordinate columns detected: **{lat_col}** (latitude), **{lon_col}** (longitude)")
                
                # Add a test feature
                with st.expander("🧪 Test Single Coordinate"):
                    st.write("Test the geocoding service with a single coordinate before processing your entire file:")
                    test_lat = st.number_input("Test Latitude", value=float(df[lat_col].iloc[0]) if not pd.isna(df[lat_col].iloc[0]) else 40.7128, format="%.6f")
                    test_lon = st.number_input("Test Longitude", value=float(df[lon_col].iloc[0]) if not pd.isna(df[lon_col].iloc[0]) else -74.0060, format="%.6f")
                    
                    if st.button("🔍 Test Geocoding"):
                        geocoder = ReverseGeocoder()
                        with st.spinner("Testing..."):
                            result = geocoder.reverse_geocode_coordinate(test_lat, test_lon)
                        
                        if result['full_address']:
                            st.success("✅ Geocoding test successful!")
                            st.write(f"**Address:** {result['full_address']}")
                            st.write(f"**City:** {result['city']}")
                            st.write(f"**State:** {result['state']}")
                            st.write(f"**Country:** {result['country']}")
                        else:
                            st.error("❌ No address found for these coordinates. Check your internet connection or try different coordinates.")
                            
            else:
                st.error("❌ Could not detect latitude/longitude columns. Please ensure your CSV has columns named: latitude/lat/y and longitude/lon/lng/long/x")
                st.stop()
            
            # Process button
            if st.button("🚀 Start Reverse Geocoding", type="primary", disabled=st.session_state.processing):
                
                # Set processing state
                st.session_state.processing = True
                
                # Initialize geocoder
                geocoder = ReverseGeocoder()
                
                try:
                    # Process the dataframe
                    with st.spinner("Processing coordinates..."):
                        result_df, successful_geocodes = geocoder.process_coordinates_dataframe(
                            df.copy(), 
                            rate_limit=rate_limit
                        )
                    
                    # Store results in session state
                    st.session_state.results = {
                        'dataframe': result_df,
                        'successful_geocodes': successful_geocodes,
                        'total_rows': len(df),
                        'original_filename': uploaded_file.name
                    }
                    
                    # Reset processing state
                    st.session_state.processing = False
                    
                    # Force a rerun to display results
                    st.rerun()
                    
                except Exception as e:
                    st.session_state.processing = False
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.info("**Troubleshooting tips:**")
                    st.write("1. Make sure your CSV has 'Latitude' and 'Longitude' columns (or similar)")
                    st.write("2. Check that coordinates are valid decimal numbers")
                    st.write("3. Ensure you have a stable internet connection")
                    st.write("4. Try with a smaller file first to test")
                    
                    # Show column debugging info
                    st.write("**Debug info:**")
                    st.write(f"Columns in your file: {list(df.columns)}")
                    st.write(f"Sample data from first row:")
                    st.write(df.head(1))
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("Please make sure your file is a valid CSV with proper formatting.")
    
    # Display results if they exist
    if st.session_state.results:
            result_df = st.session_state.results['dataframe']
            successful_geocodes = st.session_state.results['successful_geocodes']
            total_rows = st.session_state.results['total_rows']
            original_filename = st.session_state.results['original_filename']
            
            # Display results
            st.success(f"🎉 Successfully geocoded {successful_geocodes}/{total_rows} coordinates!")
            
            # Results section
            st.header("📋 Results")
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", total_rows)
            with col2:
                st.metric("Successful Geocodes", successful_geocodes)
            with col3:
                success_rate = (successful_geocodes / total_rows) * 100 if total_rows > 0 else 0
                st.metric("Success Rate", f"{success_rate:.1f}%")
            
            # Country breakdown
            if successful_geocodes > 0:
                country_counts = result_df[result_df['country'] != '']['country'].value_counts()
                if not country_counts.empty:
                    st.subheader("🌎 Locations by Country")
                    st.bar_chart(country_counts)
            
            # Display sample results
            st.subheader("🔍 Sample Results")
            sample_cols = ['full_address', 'city', 'state', 'country']
            available_cols = [col for col in sample_cols if col in result_df.columns]
            if available_cols:
                st.dataframe(result_df[available_cols].head(10), use_container_width=True)
            
            # Download section
            st.header("⬇️ Download Results")
            
            # Convert dataframe to CSV
            csv_buffer = io.StringIO()
            result_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            # Download button
            st.download_button(
                label="📥 Download Geocoded CSV",
                data=csv_data,
                file_name=f"geocoded_{original_filename}",
                mime="text/csv",
                type="primary"
            )
            
            # Display full results (optional)
            with st.expander("🗂️ View All Results"):
                st.dataframe(result_df, use_container_width=True)
            
            # Clear results button
            if st.button("🔄 Process New File"):
                st.session_state.results = None
                st.rerun()
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a CSV file to get started")
        
        # Sample data format
        st.subheader("📋 Expected CSV Format")
        sample_data = pd.DataFrame({
            'Name': ['Location 1', 'Location 2', 'Location 3'],
            'Latitude': [40.7128, 34.0522, 41.8781],
            'Longitude': [-74.0060, -118.2437, -87.6298]
        })
        st.dataframe(sample_data, use_container_width=True)


if __name__ == "__main__":
    # This script should be run with 'streamlit run app.py'
    # Running with 'python app.py' will cause ScriptRunContext warnings
    main()