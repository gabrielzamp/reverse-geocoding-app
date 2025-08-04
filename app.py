#!/usr/bin/env python3
"""
Reverse Geocoding Script - Coordinates to Addresses
This script takes a CSV with coordinates (lat/lon) and adds full address information
using reverse geocoding.
"""

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
from typing import Dict, Optional

class ReverseGeocoder:
    def __init__(self, user_agent="reverse_geocoder_tool"):
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
            print(f"  ⚠ Geocoding service error for ({latitude}, {longitude}): {e}")
            return self._empty_address_result()
        except Exception as e:
            print(f"  ✗ Error reverse geocoding ({latitude}, {longitude}): {e}")
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

    def process_coordinates_csv(self, input_file: str, output_file: str = None) -> pd.DataFrame:
        """
        Process CSV file with coordinates and add address information

        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file (optional)

        Returns:
            DataFrame with added address columns
        """
        print(f"Loading coordinates from: {input_file}")

        try:
            # Read the CSV file
            df = pd.read_csv(input_file)
            print(f"Loaded {len(df)} rows from CSV")

            # Check for required columns
            required_cols = ['Latitude', 'Longitude']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            print(f"Found columns: {list(df.columns)}")

            # Initialize new address columns
            address_columns = [
                'full_address', 'house_number', 'street', 'street_address',
                'city', 'county', 'state', 'postal_code', 'country', 'country_code'
            ]

            for col in address_columns:
                df[col] = ''

            # Process each row
            successful_geocodes = 0
            print("\nStarting reverse geocoding process...")
            print("-" * 50)

            for index, row in df.iterrows():
                latitude = row['Latitude']
                longitude = row['Longitude']

                # Skip if coordinates are missing or invalid
                if pd.isna(latitude) or pd.isna(longitude):
                    print(f"Row {index + 1}: Skipping - missing coordinates")
                    continue

                print(f"Row {index + 1}/{len(df)}: Reverse geocoding ({latitude:.6f}, {longitude:.6f})")

                # Perform reverse geocoding
                address_info = self.reverse_geocode_coordinate(latitude, longitude)

                # Update DataFrame with address information
                for col in address_columns:
                    df.at[index, col] = address_info[col]

                if address_info['full_address']:
                    successful_geocodes += 1
                    print(f"  ✓ Success: {address_info['full_address']}")
                else:
                    print(f"  ✗ No address found")

                # Rate limiting - be nice to the free service
                time.sleep(1.1)

            print("-" * 50)
            print(f"Reverse geocoding complete!")
            print(f"Successfully geocoded: {successful_geocodes}/{len(df)} coordinates")

            # Export results
            if output_file:
                df.to_csv(output_file, index=False)
                print(f"Results exported to: {output_file}")
            else:
                # Create default output filename
                base_name = input_file.replace('.csv', '')
                output_file = f"{base_name}_with_addresses.csv"
                df.to_csv(output_file, index=False)
                print(f"Results exported to: {output_file}")

            return df

        except FileNotFoundError:
            print(f"Error: Could not find input file '{input_file}'")
            raise
        except Exception as e:
            print(f"Error processing file: {e}")
            raise

    def print_sample_results(self, df: pd.DataFrame, num_samples: int = 5):
        """Print sample results for verification"""
        print(f"\nSample Results (first {num_samples} rows):")
        print("=" * 80)

        for i in range(min(num_samples, len(df))):
            row = df.iloc[i]
            print(f"\nRow {i + 1}:")
            print(f"  Original: ({row['Latitude']:.6f}, {row['Longitude']:.6f})")
            if 'Name' in df.columns:
                print(f"  Name: {row['Name']}")
            print(f"  Address: {row['full_address']}")
            print(f"  City: {row['city']}, {row['state']} {row['postal_code']}")
            print(f"  Country: {row['country']}")


def main():
    """Main execution function"""
    print("Reverse Geocoding Tool - Coordinates to Addresses")
    print("=" * 55)

    # Initialize reverse geocoder
    geocoder = ReverseGeocoder()

    # Configuration
    input_filename = "coordinates.csv"  # Change this to your input file
    output_filename = None  # Will auto-generate if None

    print(f"Input file: {input_filename}")
    print("Expected CSV columns: Name, Radius, Unit, Latitude, Longitude, LocStr, PIN")
    print("Required columns: Latitude, Longitude")
    print()

    try:
        # Process the coordinates file
        result_df = geocoder.process_coordinates_csv(input_filename, output_filename)

        # Show sample results
        geocoder.print_sample_results(result_df)

        # Print summary statistics
        print(f"\nSummary Statistics:")
        print("=" * 30)
        total_rows = len(result_df)
        successful_geocodes = len(result_df[result_df['full_address'] != ''])
        success_rate = (successful_geocodes / total_rows) * 100 if total_rows > 0 else 0

        print(f"Total coordinates processed: {total_rows}")
        print(f"Successful reverse geocodes: {successful_geocodes}")
        print(f"Success rate: {success_rate:.1f}%")

        # Country breakdown
        if successful_geocodes > 0:
            country_counts = result_df[result_df['country'] != '']['country'].value_counts()
            print(f"\nLocations by country:")
            for country, count in country_counts.items():
                print(f"  {country}: {count}")

        print("\n" + "=" * 55)
        print("REVERSE GEOCODING COMPLETE!")
        print("=" * 55)
        print("Your CSV now includes full address information.")
        print("New columns added: full_address, street_address, city, state, postal_code, country")

    except Exception as e:
        print(f"Script failed with error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your CSV file exists and has 'Latitude' and 'Longitude' columns")
        print("2. Check that coordinates are valid decimal numbers")
        print("3. Ensure you have internet connection for geocoding service")


if __name__ == "__main__":
    main()