## EPA Green Vehicle Guide (2008-2018)

1. Analyze
2. Predicative Models

### Task

### Problem 1

You as the part of project have to explore, analyze and treat the data for the below mentioned points.

- number of samples in each year
- number of columns in each dataset
- duplicate rows in each dataset
- datatypes of columns
- features with missing values
- number of non-null unique values for features in each dataset
- what those unique values are and counts for each
- analysis by grouping certain features

### Problem 2

Draw certain conclusions by creating visuals to communicate the changes in the cars in span of 10 years.

- Are more unique models using alternative fuels in 2018 compared to other years? By how much?
- How much have vechicle classes improved in fuel economy (increased in mpg) per year?
- What are the characteristics of SmartWay vechicles? Have they changed over time? (mpg, greenhouse gas)
- What features are associated with better fuel economy(mpg)?

### Problem 3

1. Also you need to merge all the datasets with same columns name by adding a columns of year to differentiate the data for different year (drop the extra columns). Also make sure that data types of merged columns are are same.
2. Find the appropriate number of clusters from the data and conclude the properties of the clusters.
3. Create a Predictive model to predict whether a vehicle is smartWay vehicle or not using the Classification Algorithm (Logistic Regression and Decisiion Tree). Also conclude, which is the best model and why not other.
4. Also, create a model to predict the City Mileage per gallon (City MPG) using the best features.

## Data Description

**Model** = vehicle make and model

**Displ** = engine displacement in liters

**Cyl** = number of engine cylinders

**Trans** = transmission type plus number of gears

- _Auto_ = Automatic
- _Man_ = Manual
- _SemiAuto_ = Semi-Automatic
- _SCV_ = Selectable Continuously Variable (e.g., CVT with paddles)
- _AutomMan_ = Automated Manual
- _AMS_ = Automated Manual-Selectable (e.g., Automated Manual with paddles)
- _Other_ = other
- _CVT_ = Continously Variable
- _CM3_ = Creeper/Manual 3-Speed
- _CM4_ = Creeper/Manual 4-Speed
- _C4_ = Creeper/Manual 4-Speed
- _C5_ = Creeper/Manual 5-Speed
- _Auto-S2_ = Semi-Automatic 2-Speed
- _Auto-S3_ = Semi-Automatic 3-Speed
- _Auto-S4_ = Semi-Automatic 4-Speed
- _Auto-S5_ = Semi-Automatic 5-Speed
- _Auto-S6_ = Semi-Automatic 6-Speed
- _Auto-S7_ = Semi-Automatic 7-Speed

**Drive** = 2-wheel Drive, 4-wheel Drive/all-wheel Drive

**Fuel** = fuel(s)

**Cert Region**

- _CA_ = California
- _CE_ = Calif. + NLEV (Northeast trading area)
- _CF_ = Clean Fuel Vehicle
- _CL_ = Calif. + NLEV (All states)
- _FA_ = Federal All Altitude
- _FC_ = Tier 2 Federal and Calif.
- _NF_ = CFV + NLEV(ASTR) + Calif.
- _NL_ = NLEV (All states)

**Stnd** = vehile emissions standard code.

**[Stnd Description](https://www.epa.gov/greenvehicles/federal-and-california-light-duty-vehicle-emissions-standards-air-pollutants)** = vehicle emissions standard description.

**[Underhood ID](http://www.fueleconomy.gov/feg/findacarhelp.shtml#airPollutionScore)** = Engine family or test group ID.

**[Veh Class](http://www.fueleconomy.gov/feg/findacarhelp.shtml#epaSizeClass)** = EPA vehicle class.

**[Air Pollution Score](http://www.fueleconomy.gov/feg/findacarhelp.shtml#airPollutionScore)** = Smog Rating[?](https:/www.epa.gov/greenvehicles/smog-rating)

**City MPG** = City fuel economy in miles per gallon

**Hwy MPG** = highway fuel economy in miles per gallon

**Cmd MPG** = combined city/highway fuel economy in miles per gallon

**[Greenhouse Gas Score](https://www.epa.gov/greenvehicles/greenhouse-gas-rating)** = Greenhouse Gas Rating

**[SmartWay](https://www.epa.gov/greenvehicles/consider-smartway-vehicle)** = Yes, No, or Elite.

**Comb C0<sub>2</sub>** = Combined city/highway CO<sub>2</sub> tailpipe emissions in grams per mile.
​

## Reference

1. [https://pythonplot.com/](https://pythonplot.com/)
2. [https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/](https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/)

## Todo

- [ ] Number of Samples in each Year

- [ ] Number of Columns in each dataset

- [ ] Duplicate rows in each dataset

- [ ] Datatypes of Columns

- [ ] Features with missing values

- [ ] numbers of non-null unique values for feature in each dataset

- [ ] what those unique values are and counts for each

- [ ] analysis by grouping certain features
