# DstIndex ( Space Weather ) Prediction using Pre Trained LSTM Model
Contains data cleaning and LSTM model code for dst index predictor.

This Python script leverages Selenium, concurrent.futures, and TensorFlow to predict space weather conditions based on real-time data. The script collects data from two sources:

NASA Wind Data: Scrapes real-time solar wind data from [NASA's MFI SWE](https://wind.nasa.gov/mfi_swe_plot.php) Plot page.
SIDC Solar Activity Data: Retrieves the latest solar activity data from the [SIDC - Solar Influences Data Analysis Center](https://www.sidc.be/SILSO/DATA/EISN/EISN_current.txt).
The collected data is then used to make predictions using a pre-trained machine learning model saved in "model.h5."

## Prerequisites
Ensure you have the necessary libraries installed:
```bash
pip install selenium numpy tensorflow
```
Additionally, make sure you have the ChromeDriver installed and the model.h5 file available.
## Execution
1. Run the script using:
```bash
python realtimedst.py
```
The script utilizes Selenium with a headless Chrome browser to scrape data concurrently from the mentioned URLs.

The scraped data is then used to make predictions using a pre-trained machine learning model.

The results are stored in variables, including solar wind speed, density, and various magnetic field components.


## Configuration
Adjust the URLs in process_url1 and process_url2 functions if the data sources change.
Ensure that the ChromeDriver path is correctly set.
Make sure the model.h5 file is present and corresponds to the expected input dimensions.
## Note
This script is designed for educational purposes and requires consistent internet access to fetch real-time data. Adjustments may be needed based on changes to the data sources or page structures. I have additionally added the notebook used for data processing to obtain model.h5 and also links to download in this notebook. 
