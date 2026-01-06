# nyc-houses-eda-and-technical-data-problems
originally wanted to create a price estimator for NYC houses, I faced problems and wasn't able to develop an accurate model. so  this project became about finding a reason why, studying the dataset and elaborate possible explanations

I proceed to upload the dataframe and perform an EDA (exploratory data analysis) of it. Data looks quite complete and good quality, but the heatmap of correlation looks odd. In particular, the price column seems to have quite low correlation with the others. Also, the great majority of my data has a sale price below the average as it is shown by the shape of the graph skewed on the left. 
<img width="480" height="400" alt="image" src="https://github.com/user-attachments/assets/69a5571b-337d-4614-88ff-e2f690162d08" /><img width="459" height="450" alt="image" src="https://github.com/user-attachments/assets/6ca33b95-960f-44c0-8b61-5ee63f00fce1" /> 

After the EDA I start to train my first model (MLPRegressor) but it perform quite bad (considerable standard error and a R-squared coefficient of only 32%). I then try to change my algorithm and try to use a random forest. It goes better but still my model can't explain more than 50% of the total variability of the price.
I tried to find a solution and asked an opinion into some specialized groups, the general opinion was that the problem was not in the algorithm but in the data.

## So what can I do to try to improve the model?

- do some data engeneering
- tuning my model
- drop some columns

