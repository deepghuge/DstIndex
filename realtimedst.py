from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import concurrent.futures
import numpy as np
import tensorflow as tf
from tensorflow import keras 

chrome_options = Options()
chrome_options.add_argument("--headless")

def process_url1():
    with webdriver.Chrome(options=chrome_options) as driver1:
        driver1.get("https://wind.nasa.gov/mfi_swe_plot.php")
        radio_start = driver1.find_element(By.CSS_SELECTOR, "#contentwrapper > center > h4 > form > input[type=radio]:nth-child(2)").click()
        
        
        driver1.find_element(By.CSS_SELECTOR,"#contentwrapper > center > h4 > form > input[type=radio]:nth-child(30)").click()
        speed_mean = driver1.find_element(By.CSS_SELECTOR,"#contentwrapper > center > h4 > form > input[type=checkbox]:nth-child(58)").click()
        density_mean = driver1.find_element(By.CSS_SELECTOR,"#contentwrapper > center > h4 > form > input[type=checkbox]:nth-child(70)").click()
     
        submit = driver1.find_element(By.CSS_SELECTOR, "#contentwrapper > center > h4 > form > input[type=submit]:nth-child(98)").click()
        txt_data1 = driver1.find_element(By.CSS_SELECTOR, "body > pre").text
        lines1 = txt_data1.split('\n')
        last_row1 = lines1[-1]
        nn1 = np.array(last_row1.split())
        return nn1

def process_url2():
    with webdriver.Chrome(options=chrome_options) as driver2:
        driver2.get("https://www.sidc.be/SILSO/DATA/EISN/EISN_current.txt")
        txt_data2 = driver2.find_element(By.CSS_SELECTOR, "body > pre").text
        lines2 = txt_data2.split('\n')
        last_row2 = lines2[-1]
        nn2 = np.array(last_row2.split())
        return nn2


with concurrent.futures.ThreadPoolExecutor() as executor:
    result1 = executor.submit(process_url1)
    result2 = executor.submit(process_url2)

# Get the results
nn1 = result1.result()
nn2 = result2.result()

bt_mean = float(nn1[3])
bt_std = -0.001599576
temperature_mean = 0.026983501
temperature_std = 0.0248886813
bx_gse_mean = float(nn1[4])
bx_gse_std = 0.0137303226
by_gse_mean = float(nn1[5])
by_gse_std = 0.008941557
bz_gse_mean = float(nn1[6])
bz_gse_std = 0.0156987311
speed_mean = float(nn1[7])
speed_std = 0.0247443397
density_mean = float(nn1[8])
density_std = 0.01092404
smoothed_ssn = float(nn2[4])

model = keras.models.load_model("model.h5")
test = np.array([[[bt_mean,bt_std,temperature_mean,temperature_std,bx_gse_mean,bx_gse_std,by_gse_mean,by_gse_std,bz_gse_mean,bz_gse_std,speed_mean,speed_std,density_mean,density_std,smoothed_ssn]]])
test_reshaped = np.repeat(test, 34, axis=1)
predictions = model.predict(test_reshaped)

