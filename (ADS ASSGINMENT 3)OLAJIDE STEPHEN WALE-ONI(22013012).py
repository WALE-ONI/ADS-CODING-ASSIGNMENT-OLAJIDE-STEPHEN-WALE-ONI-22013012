# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:09:46 2023

@author: OLAJIDE STEPHEN WALE-ONI(22013012)
"""

import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import seaborn as sns
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import cluster_tools as ct
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit

# Creating a def function to read in data

# Creating a def function to read in data

"""
Creating a def function to read in our datasets and skiprows the first 4 rows
"""
def read_data(filename, **others):
    """
    A function that reads in a dataset and skips the first 4 rows.

    Args:
        filename (str): The name or path of the file to be read.
        **kwargs: Additional keyword arguments to be passed to pd.read_csv.

    Returns:
        pandas.DataFrame: The dataset read from the file with the first 4 rows skipped.
    """
    world_data = pd.read_csv(filename, skiprows=4)

    return GDP_capita


#Reading the data frame
GDP_capita = pd.read_csv("CO2perGDP.csv", skiprows=4)
GDP_capita.describe()
  
# Selecting the countries needed
GDP_capita = GDP_capita[GDP_capita['Country Name'].isin(['Senegal', 'Sri Lanka', 'Kenya', 'France', 'Algeria', 'Suriname', 'Ghana', 'United States', 'Malawi', 'Papua New Guinea', 'Nepal', 'Colombia', 'Seychelles', 'Lesotho', 'Philippines', 'Puerto Rico', 'Mexico', 'Singapore', 'Japan', "Cote d'Ivoire", 'Fiji', 'Rwanda', 'Hong Kong SAR, China', 'Australia', 'Congo, Dem. Rep.', 'St. Vincent and the Grenadines', 'Sweden', 'Portugal', 'United Kingdom'])]
print(GDP_capita)

# Dropping the columns not needed
GDP_capita = GDP_capita.drop(['Indicator Name', 'Country Code', 'Indicator Code'], axis=1)
print(GDP_capita)


# reseting the index
GDP_capita.reset_index(drop=True, inplace=True)
print(GDP_capita)


# Extracting  years from our urban out dataset

GDP_capita_s=GDP_capita[['Country Name','1990', '2000','2010', '2019']]
GDP_capita_s.describe()

# Checking for missing values
GDP_capita_s.isna().sum()


# Transposing the data
GDP_capita_t = GDP_capita_s.T

GDP_capita_t.columns = GDP_capita_t.iloc[0]
GDP_capita_t = GDP_capita_t.iloc[1:]
GDP_capita_t.describe()
GDP_capita_t = GDP_capita_t.apply(pd.to_numeric)
GDP_capita_t.plot()



# In this data, i will be working with 2019 data
# Extracting 30 years of data at an interval of 10 years from out dataset
GDP_capita_year=GDP_capita_s[['1990', '2000','2010', '2019']]
GDP_capita_year.describe()

# Checking for missing values
GDP_capita_year.isna().sum()

# dropping the missing values
GDP_capita_year.dropna(inplace=True)



# Checking for correlation between our years choosen
# Correlation
corr = GDP_capita_year.corr()
#print(corr)


#Plotting a Heatmap
sns.heatmap(corr, cmap='coolwarm', annot=True)
plt.title("Correlation Heatmap")
plt.show()

# plotting a scatter plot
pd.plotting.scatter_matrix(GDP_capita_year, figsize=(15, 15))
plt.tight_layout()    # helps to avoid overlap of labels
plt.show()


#creating clusters
GDP_capita_cluster=GDP_capita_year[['1990', '2019']].copy()
GDP_capita_cluster


# Normalizing the data and storing minimum and maximum value

GDP_capita_norm, GDP_capita_min, GDP_capita_max = ct.scaler(GDP_capita_cluster)
print(GDP_capita_norm.describe())


# Calculating the best clustering number

for i in range(2, 9):
    # creating  kmeans and fit
    kmeans = cluster.KMeans(n_clusters=i)
    kmeans.fit(GDP_capita_cluster)     

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (i, skmet.silhouette_score(GDP_capita_cluster, labels))
    
    
   # 2 and 3 has the highest silhoutte score respectively , so i will be plotting for 2 and 3 clusters and choosing the best
 
    # Plotting for 3 clusters
nclusters = 3 # number of cluster centres

kmeans = cluster.KMeans(n_clusters=nclusters)
kmeans.fit(GDP_capita_norm)     

# extract labels and cluster centres
labels = kmeans.labels_

# extracting the estimated number of cluster
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))
# scatter plot with colours selected using the cluster numbers
plt.scatter(GDP_capita_norm["1990"], GDP_capita_norm["2019"], c=labels, cmap="tab10")
# colour map Accent selected to increase contrast between colours

# show cluster centres
xcen = cen[:,0]
ycen = cen[:,1]
plt.scatter(xcen, ycen, c="k", marker="d", s=80)
# c = colour, s = size

plt.xlabel(" GDP_capita(1990)")
plt.ylabel(" GDP_capita_norm(2019)")
plt.title("3clusters")
plt.show()


# Scaling back to the original data and creating a plot it on the original scale

plt.style.use('seaborn')
plt.figure(dpi=300)

# now using the original dataframe
plt.scatter(GDP_capita["1990"], GDP_capita["2020"], c=labels, cmap="tab10")


# rescale and show cluster centres
scen = ct.backscale(cen, GDP_capita_min, GDP_capita_max)
xc = scen[:,0]
yc = scen[:,1]
plt.scatter(xc, yc, c="k", marker="d", s=80)

plt.xlabel("1990")
plt.ylabel("2020")
plt.title("GDP PER CAPITA")
plt.show()



#Transpossing
GDP_capita_trans = GDP_capita.T

# Making the country name the colums
GDP_capita_trans.columns=GDP_capita_trans.iloc[0]
GDP_capita_trans



# Selecting only the years the data frame
GDP_capita_years=GDP_capita_trans.iloc[1:]
GDP_capita_years


GDP_capita_years=GDP_capita_years.apply(pd.to_numeric)
GDP_capita_years


# resetting the index
GDP_capita_years.reset_index(inplace=True)
GDP_capita_years

#cheching the values of fiji and years
GDP_capita_years['Fiji'].values
GDP_capita_years['Year'].values

#renaming the country name to years
GDP_capita_years.rename(columns={'index': 'Year'}, inplace=True)
GDP_capita_years

GDP_capita_years.dtypes

GDP_capita_years['Year'] = GDP_capita_years['Year'].astype('int')
GDP_capita_years

GDP_capita_years.dtypes
GDP_capita_years
GDP_capita_years.columns





# Exponential function for Fiji
def exponential(t, a, b):
    """Computes exponential growth
    
    Parameters:
        t: The current time
        a: The initial value
        b: The growth rate
        
    Returns:
        The value at the given time
    """
    return a * np.exp(b * t)

years = GDP_capita_years['Year'].values
GDP = GDP_capita_years['Fiji'].values

# Provide initial guess for exponential function
initial_guess = [min(GDP), 0.01]  # You can adjust the initial guess if needed

try:
    popt, pcov = curve_fit(exponential, years, GDP, p0=initial_guess, maxfev=10000)
except RuntimeError as e:
    print("Curve fitting failed:", str(e))
    popt = initial_guess

# Generate points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
curve_GDP = exponential(curve_years, *popt)

# Predictions for 2030 and 2040
prediction_years = np.array([2030, 2040])
predictions = exponential(prediction_years, *popt)
print("GDP per capita prediction for 2030:", predictions[0])
print("GDP per capita prediction for 2040:", predictions[1])

# Plot the data, fitted curve, and predictions
plt.plot(years, GDP, 'ro', label='Data')
plt.plot(curve_years, curve_GDP, 'b-', label='Fitted Curve')
plt.plot(prediction_years, predictions, 'g*', label='Predictions')
plt.plot([curve_years[-1], prediction_years[0]], [curve_GDP[-1], predictions[0]], 'g--')
plt.plot([curve_years[-1], prediction_years[1]], [curve_GDP[-1], predictions[1]], 'g--')
plt.xlabel('Year')
plt.ylabel('GDP per capita')
plt.title('Exponential Growth Fit for GDP per capita of Fiji')
plt.legend()
plt.grid(True)
plt.show()




#fitting for United States using Polynomial function


def polynomial(t, *coefficients):
    """Computes a polynomial function
    
    Parameters:
        t: The current time
        coefficients: Coefficients of the polynomial function
        
    Returns:
        The value at the given time
    """
    return np.polyval(coefficients, t)

# Obtain the years and GDP data
years = GDP_capita_years['Year'].values
GDP = GDP_capita_years['United States'].values

# Define the degree of the polynomial
degree = 3

# Perform polynomial curve fitting
coefficients = np.polyfit(years, GDP, degree)

# Generate points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
curve_GDP = polynomial(curve_years, *coefficients)

# Predictions for 2030 and 2040
prediction_years = np.array([2030, 2040])
predictions = polynomial(prediction_years, *coefficients)
print("GDP per capita prediction for 2030:", predictions[0])
print("GDP per capita prediction for 2040:", predictions[1])

# Plot the data, fitted curve, and predictions
plt.plot(years, GDP, 'ro', label='Data')
plt.plot(curve_years, curve_GDP, 'b-', label='Fitted Curve')
plt.plot(prediction_years, predictions, 'g*', label='Predictions')
plt.plot([curve_years[-1], prediction_years[0]], [curve_GDP[-1], predictions[0]], 'g--')
plt.plot([curve_years[-1], prediction_years[1]], [curve_GDP[-1], predictions[1]], 'g--')
plt.xlabel('Year')
plt.ylabel('GDP per capita')
plt.title('GDP per capita Polynomial Fitting of United States')
plt.legend()
plt.grid(True)
plt.show()




#fitting for Colombia using Polynomial function


def polynomial(t, *coefficients):
    """Computes a polynomial function
    
    Parameters:
        t: The current time
        coefficients: Coefficients of the polynomial function
        
    Returns:
        The value at the given time
    """
    return np.polyval(coefficients, t)

# Obtain the years and GDP data
years = GDP_capita_years['Year'].values
GDP = GDP_capita_years['Colombia'].values

# Define the degree of the polynomial
degree = 3

# Perform polynomial curve fitting
coefficients = np.polyfit(years, GDP, degree)

# Generate points for the fitted curve
curve_years = np.linspace(min(years), max(years), 100)
curve_GDP = polynomial(curve_years, *coefficients)

# Predictions for 2030 and 2040
prediction_years = np.array([2030, 2040])
predictions = polynomial(prediction_years, *coefficients)
print("GDP per capita prediction for 2030:", predictions[0])
print("GDP per capita prediction for 2040:", predictions[1])

# Plot the data, fitted curve, and predictions
plt.plot(years, GDP, 'ro', label='Data')
plt.plot(curve_years, curve_GDP, 'b-', label='Fitted Curve')
plt.plot(prediction_years, predictions, 'g*', label='Predictions')
plt.plot([curve_years[-1], prediction_years[0]], [curve_GDP[-1], predictions[0]], 'g--')
plt.plot([curve_years[-1], prediction_years[1]], [curve_GDP[-1], predictions[1]], 'g--')
plt.xlabel('Year')
plt.ylabel('GDP per capita')
plt.title('GDP per capita Polynomial Fitting of Colombia')
plt.legend()
plt.grid(True)
plt.show()























