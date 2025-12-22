# Housing Price Prediction Model with pytorch

Simple model built using pytorch, data cleaned and prepped using pandas, numpy. I will be using Selenium to
scrape for testing data to get a more expansive evaluation of the model.

The goal of this project is to create an accurate predictor of housing prices based solely on GPS coordinates.
I quickly expanded this idea, as coordinates alone was undermining the potential of the model. I also need to
switch datasets, as I realized too late that my original data did not have any price data, which made it unusable
for my goal. What this process lacked in cleanliness,  it made up for in learning experiences. It highlighted the 
importance of data visualization and interpretation before starting a project, and secured a lot of fundamentals 
of how models should be set up and tuned.

I have migrated over to a different dataset, which has the features I need. The model is sitting at a normalized
RMSE of ~ 0.6, which equates to about a 0.5 - 1.5 times price variation. I plan to expand the features I am using
to make the model more optimal.
