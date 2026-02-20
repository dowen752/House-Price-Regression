# Housing Price Prediction Model with pytorch

Simple model to predict housing prices using from latitude, longitude, lot area, number of bedrooms, number of bathrooms,
along with derived features like floorplan density and distance to the nearest metropolitan area. Built using pytorch,
data cleaned and prepped using pandas, numpy. The model is currently at a log-price R^2 value of 0.637 (i.e, my model 
reduces prediction error variance by 64%, compared to just using the mean price to predict). My RMSE is ~$400k, which 
is high, but is something I expected for a model of this simplicity.

The goal of this project was to create an accurate predictor of housing prices based solely on GPS coordinates.
I quickly expanded this idea, as coordinates alone was undermining the potential of the model. I also need to
switch datasets, as I realized too late that my original data did not have any price data, which made it unusable
for my goal. What this process lacked in cleanliness,  it made up for in learning experiences. It highlighted the 
importance of data visualization and interpretation before diving into a project, and secured a lot of fundamentals 
of how models should be set up and tuned.

I have migrated over to a different dataset, which has the features I need. Although this is performing much better, 
there is a lot of variance that my model still doesn't explain. This makes sense though, as houses are not just the 
summation of a short list of features. You can have all the same figures but one house has a bad layout, is far from 
any shopping areas, was built many years ago. I wanted to build a model that has relative ease of access, so I don't
plan to expand features from here, though it definitely could be.
