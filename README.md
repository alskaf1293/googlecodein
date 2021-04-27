Requirements:
Tensorflow
Quandl
Sklearn
Pandas

4/26/2021 edit: This is an old project that I wrote for a Google CodeIn competition a couple of years back. There aren't any demonstrations, but this was supposed to be a stock trading algorithm. Anyways, I guess you could look at this to take a look my tensorflow/data-preprocessing knowledge, but the dependencies are probably so outdated that it probably wouldn't even be worth cleaning up.

This is a completed task for adding an experiment to Tensorboard.dev

In order to consistently run this code, on must have an API key that is registered in Quandl's database.

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
