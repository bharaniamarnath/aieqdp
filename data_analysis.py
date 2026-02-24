import pandas as pd
from io import StringIO

def analyze_data(data_values):
    import pandas as pd
    # Initialize a buffer for info
    buffer = StringIO()

    # Display the first few rows of the dataset
    head = f"Dataset Sample:\n{data_values.head()}\n"

    # Get dataset information
    data_values.info(buf=buffer)
    info_buffer = f"Dataset Information:\n{buffer.getvalue()}\n"
    buffer.truncate(0)  # Clear the buffer for the next use
    buffer.seek(0)

    # Description
    desc_buffer = f"Dataset Description:\n{data_values.describe().to_string()}\n"

    # Check for missing values
    missing_values = data_values.isnull().sum()
    if missing_values.any():
        missing_details = missing_values[missing_values > 0].to_string()  # Convert to string
    else:
        missing_details = "No missing values in the dataset."

    return str(head), str(info_buffer), str(desc_buffer), str(missing_details)

