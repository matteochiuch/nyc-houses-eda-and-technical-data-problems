# nyc-houses-eda-and-technical-data-problems
originally wanted to create a price estimator for NYC houses, I faced problems and wasn't able to develop an accurate model. so  this project became about finding a reason why, studying the dataset and elaborate possible explanations

I proceed to upload the dataframe and perform an EDA (exploratory data analysis) of it. Data looks quite complete and good quality, but the heatmap of correlation looks odd. In particular, the price column seems to have quite low correlation with the others. Also, the great majority of my data has a sale price below the average as it is shown by the shape of the graph skewed on the left. Furthermore, the scatterplot reveals the absence of a defined trend and a considerable dispersion of the data.

<img width="430" height="400" alt="image" src="https://github.com/user-attachments/assets/69a5571b-337d-4614-88ff-e2f690162d08" /><img width="419" height="450" alt="image" src="https://github.com/user-attachments/assets/6ca33b95-960f-44c0-8b61-5ee63f00fce1" /> 

After the EDA I start to train my first model (MLPRegressor) but it perform quite bad (considerable standard error and a R-squared coefficient of only 32%). I then try to change my algorithm and try to use a random forest. It goes better but still my model can't explain more than 50% of the total variability of the price.
I tried to find a solution and asked an opinion into some specialized groups, the general opinion was that the problem was not in the algorithm but in the data.

## So what can I do to try to improve the model?

- **Data engeneering**

I tried to create some new columns from the ones I already have combining some data in order to get new ratios. Moreover, I also tried to split the categorical variables such as building class to get some dummy's columns and create a new dummy for the commercial area of the buildings
- **tuning my model**

I run some tests to see what kind of hyperparameters would perform better. In particular with 3 different numbers of estimator, 4 different depth and 3 different minimum sample split numbers, this created 36 different models, of which I choose the best one
  
- **drop some columns**

I selected the variables I tought to be the most relevant and I dropped the others in an attempt to simplify the model. This didn't actually improved the model but neither it has made it worse so the variables dropped were actually kinda useless to estimate the price.

## general conclusion

I have been able to develop a prediction model, but its performance are actually quite disappointing. Even studying the data their quality seems good and usual basic techniques to handle this kind of problems has revealed to be ineffective.

- One possible explanation of this behavior is that the dataset misses some relevant feature for this estimation, feature that actually account for almost 50% of the total variability of the price.
- Another possibility is that the dataset is actually complete and the price of the houses in New York follows some quite randomic pattern, thus it is actually unpredictable (which seems not logical nor likely in my opinion).
- Finally, there is the possibility that these data are synthetic: since this was an online dataset created for training purpouses, the data contained lack some grade of contact with reality. In particular, while we can actually observe multicollinearity of some columns (for example building area and lot area) that is not given for the price as if the price of the houses is only slightly dependent from the variables available. Since also the other works on this dataset are mostly EDAs and I wasn't able to find any regression model, it is likely to think that also other people has incurred in this same problems and, if so, this could indicate a problem with the dataset itself.


  

