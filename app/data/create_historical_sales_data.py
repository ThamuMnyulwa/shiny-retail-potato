import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import math


def generate_retail_data(
    start_date="2025-01-01",
    num_weeks=10,
    countries=["Nigeria", "Zimbabwe"],
    branches_per_country=2,
    skus=[
        {"sku_no": "A001", "sku_name": "Premium Coffee Blend"},
        {"sku_no": "B002", "sku_name": "Organic Green Tea"},
    ],
    distribution_type="lognormal",  # Options: lognormal, gamma, weibull, custom
    noise_level=0.15,  # Scale of noise (0.0 to 1.0)
    seasonality_factor=0.1,  # How much sales increase month-to-month (0.1 = 10%)
    promotion_frequency=0.05,  # Probability of promotion on any given day
    promotion_boost_range=(1.3, 1.8),  # Min/max multiplier for promotions
    day_of_week_effects=[1.0, 0.9, 0.95, 1.0, 1.1, 1.3, 1.2],  # Mon-Sun factors
    base_sales=None,  # Can provide custom base sales per country and SKU
    output_file="custom_retail_sales.csv",
):
    """
    Generate simulated retail sales data with various distribution options

    Args:
        start_date: First date of the simulation (string YYYY-MM-DD)
        num_weeks: Number of weeks to simulate
        countries: List of countries to include
        branches_per_country: Number of branches per country
        skus: List of SKU dictionaries with sku_no and sku_name
        distribution_type: Type of distribution for the base sales variation
        noise_level: Amount of noise to add (0.0 to 1.0)
        seasonality_factor: Monthly increase factor
        promotion_frequency: Probability of a promotion on any day
        promotion_boost_range: (min, max) boost for promotions
        day_of_week_effects: List of 7 factors for days of week
        base_sales: Optional dict of {country: {sku_no: base_sales}}
        output_file: CSV file to output

    Returns:
        DataFrame of generated sales data
    """
    # Convert start date
    current_date = datetime.strptime(start_date, "%Y-%m-%d")

    # Find the next Friday (for first week ending)
    days_until_friday = (4 - current_date.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7  # If already Friday, go to next Friday
    first_friday = current_date + timedelta(days=days_until_friday)

    # Branch coordinates (specific cities in each country)
    branch_info = {
        "Nigeria": [
            {"name": "Lagos", "latitude": 6.5244, "longitude": 3.3792},
            {"name": "Abuja", "latitude": 9.0765, "longitude": 7.3986},
            {"name": "Kano", "latitude": 12.0022, "longitude": 8.5920},
            {"name": "Ibadan", "latitude": 7.3775, "longitude": 3.9470},
        ],
        "Zimbabwe": [
            {"name": "Harare", "latitude": -17.8252, "longitude": 31.0335},
            {"name": "Bulawayo", "latitude": -20.1700, "longitude": 28.5800},
            {"name": "Chitungwiza", "latitude": -18.0126, "longitude": 31.0751},
            {"name": "Mutare", "latitude": -18.9708, "longitude": 32.6709},
        ],
    }

    # Default base sales if not provided
    if base_sales is None:
        base_sales = {
            "Nigeria": {"A001": 120, "B002": 80},
            "Zimbabwe": {"A001": 90, "B002": 95},
        }

    # Create branch numbers (first digit is country code, last two are sequential)
    branch_numbers = {}
    for i, country in enumerate(countries, 1):
        branch_numbers[country] = [
            int(f"{i}0{j+1}") for j in range(branches_per_country)
        ]

    # Create data container
    data = []

    # Function to generate sales values based on the chosen distribution
    def generate_sales_value(base, noise_scale=noise_level):
        if distribution_type == "lognormal":
            # Lognormal distribution (right-skewed)
            sigma = math.sqrt(math.log(1 + noise_scale**2))
            noise_factor = np.random.lognormal(mean=0, sigma=sigma)
            return max(0, base * noise_factor)

        elif distribution_type == "gamma":
            # Gamma distribution (right-skewed, but different shape than lognormal)
            shape = 1 / noise_scale  # Higher shape = less variance
            scale = noise_scale * base
            return np.random.gamma(shape, scale)

        elif distribution_type == "weibull":
            # Weibull distribution
            shape = 1.5  # Shape parameter (k)
            scale = base  # Scale parameter (lambda)
            return np.random.weibull(shape) * scale

        elif distribution_type == "custom":
            # Custom distribution: mixture of normal and occasional spikes
            if random.random() < 0.1:  # 10% chance of higher sales
                return (
                    base
                    * random.uniform(1.2, 1.5)
                    * (1 + random.random() * noise_scale)
                )
            else:
                return base * (1 + (random.random() - 0.5) * noise_scale)

        else:
            # Default to normal noise as fallback
            noise_factor = 1 + np.random.normal(0, noise_scale)
            return max(0, base * noise_factor)

    # Generate data week by week
    current_friday = first_friday
    for week in range(1, num_weeks + 1):
        # Determine month
        month = current_friday.strftime("%B")

        # Calculate month number (for seasonality)
        month_num = (
            current_friday.month - datetime.strptime(start_date, "%Y-%m-%d").month
        )

        # Process each country and branch
        for country in countries:
            for branch_idx, branch_no in enumerate(branch_numbers[country]):
                # Get branch coordinates
                branch_coords = branch_info[country][
                    branch_idx % len(branch_info[country])
                ]

                # Generate data for each SKU
                for sku in skus:
                    sku_no = sku["sku_no"]
                    sku_name = sku["sku_name"]

                    # Get base sales for this country/SKU combination
                    base = base_sales[country][sku_no]

                    # Apply seasonality (increasing sales over months)
                    seasonal_base = base * (1 + seasonality_factor * month_num)

                    # Generate daily sales
                    daily_sales = []
                    daily_date = current_friday - timedelta(
                        days=6
                    )  # Start from Saturday for the week

                    for day in range(7):
                        # Apply day-of-week effect
                        day_base = seasonal_base * day_of_week_effects[day]

                        # Apply the chosen distribution and noise
                        daily_sale = generate_sales_value(day_base)

                        # Special events/promotions
                        if random.random() < promotion_frequency:
                            promotion_boost = random.uniform(
                                promotion_boost_range[0], promotion_boost_range[1]
                            )
                            daily_sale *= promotion_boost

                        daily_sales.append(
                            {
                                "date": daily_date.strftime("%Y-%m-%d"),
                                "sales": round(daily_sale, 2),
                            }
                        )

                        daily_date += timedelta(days=1)

                    # Calculate weekly total, opening and closing sales
                    weekly_total = sum(day["sales"] for day in daily_sales)
                    opening_sales = daily_sales[0]["sales"]
                    closing_sales = daily_sales[-1]["sales"]

                    # Add row to dataset
                    data.append(
                        {
                            "month": month,
                            "week": current_friday.strftime("%Y-%m-%d"),
                            "branch_no": branch_no,
                            "latitude": branch_coords["latitude"],
                            "longitude": branch_coords["longitude"],
                            "sku_no": sku_no,
                            "sku_name": sku_name,
                            "country": country,
                            "weekly_sales": round(weekly_total, 2),
                            "opening_sales": round(opening_sales, 2),
                            "closing_sales": round(closing_sales, 2),
                        }
                    )

        # Move to next Friday
        current_friday += timedelta(days=7)

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Write to CSV if output file is specified
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Data written to {output_file}")

    return df


# Example usage
if __name__ == "__main__":
    # Example 1: Default parameters with lognormal distribution
    df1 = generate_retail_data(
        start_date="2025-01-01", output_file="retail_sales_lognormal.csv"
    )

    # Example 2: Gamma distribution with higher noise
    df2 = generate_retail_data(
        start_date="2025-01-01",
        distribution_type="gamma",
        noise_level=0.25,
        output_file="retail_sales_gamma.csv",
    )

    # Example 3: Custom settings with Weibull distribution
    custom_base_sales = {
        "Nigeria": {"A001": 150, "B002": 70},
        "Zimbabwe": {"A001": 80, "B002": 110},
    }

    df3 = generate_retail_data(
        start_date="2025-01-01",
        num_weeks=12,
        distribution_type="weibull",
        noise_level=0.2,
        seasonality_factor=0.15,
        promotion_frequency=0.08,
        base_sales=custom_base_sales,
        output_file="retail_sales_custom.csv",
    )

    print("Data generation complete!")

    # Display sample of the data
    print("\nSample data (first 5 rows):")
    print(df3.head(5))
