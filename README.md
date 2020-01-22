This is a completed task for adding an experiment to Tensorboard.dev

main.py is the Python script that contains the code that retrieves the information from Quandl
and has the Tensorflow model.

Basically, what the model does is that it takes the information recieved from Quandl's server and calculates
percent changes, along with the Adjusted Close of the same day as a feature vector. It is preprocessed with
sklearn.preprocessing's "Scale" method. In reality, this reduces the great amount of noise that comes with
interweekly stock values (or just any stock in general) to practically nothing, which may be part of the reason why it 
works so well. I am speculating that it will not work as well as the model predicts in real time, but it is
certainly made so that it will do better than 50 percent, if needed for profit.

Regardless, this model can be retrieved from other APIs, notably AlphaVantage (retrieves intraday values) and can be
modified such that it predicts more than one day into the future, of course by modifying the variables.
