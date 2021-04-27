from qq_training_wheels.momentum_trading import MomentumTradingParams
from backtester.trading_system import TradingSystem
from backtester.features.feature import Feature
import numpy as np


class MyTradingFunctions():

    def __init__(self):
        self.count = 0
        # When to start trading
        self.start_date = '2015/01/02'
        # When to end trading
        self.end_date = '2017/08/31'
        self.params = {}

    def getSymbolsToTrade(self):
        '''
        Specify the stock names that you want to trade.
        '''
        return ['AAPL']

    def getInstrumentFeatureConfigDicts(self):
        '''
        Specify all Features you want to use by creating config dictionaries.
        Create one dictionary per feature and return them in an array.

        Feature config Dictionary have the following keys:

        featureId: a str for the type of feature you want to use
        featureKey: {optional} a str for the key you will use to call this feature
                    If not present, will just use featureId
        params: {optional} A dictionary with which contains other optional params if needed by the feature

        msDict = {
            'featureKey': 'ms_5',
            'featureId': 'moving_sum',
            'params': {
                'period': 5,
                'featureName': 'basis'
            }
        }

        return [msDict]

        You can now use this feature by in getPRediction() calling it's featureKey, 'ms_5'
        '''

        ma1Dict = {
            'featureKey': 'ma_90',
            'featureId': 'moving_average',
            'params': {
                'period': 90,
                'featureName': 'adjClose'
            }
        }
        mom30Dict = {
            'featureKey': 'mom_30',
            'featureId': 'momentum',
            'params': {
                'period': 30,
                'featureName': 'adjClose'
            }
        }
        mom10Dict = {
            'featureKey': 'mom_10',
            'featureId': 'momentum',
            'params': {
                'period': 10,
                'featureName': 'adjClose'
            }
        }
        
        return [ma1Dict, mom10Dict, mom30Dict]

    def getPrediction(self, time, updateNum, instrumentManager, predictions):
        '''
        Combine all the features to create the desired predictions for each stock.
        'predictions' is Pandas Series with stock as index and predictions as values
        We first call the holder for all the instrument features for all stocks as
            lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()
        Then call the dataframe for a feature using its feature_key as
            ms5Data = lookbackInstrumentFeatures.getFeatureDf('ms_5')
        This returns a dataFrame for that feature for ALL stocks for all times upto lookback time
        Now you can call just the last data point for ALL stocks as
            ms5 = ms5Data.iloc[-1]
        You can call last datapoint for one stock 'ABC' as
            value_for_abs = ms5['ABC']

        Output of the prediction function is used by the toolbox to make further trading decisions and evaluate your score.
        '''

        self.updateCount() # uncomment if you want a counter

        # holder for all the instrument features for all instruments
        lookbackInstrumentFeatures = instrumentManager.getLookbackInstrumentFeatures()
        
        def hurst_f(input_ts, lags_to_test=20):  
            # interpretation of return value
            # hurst < 0.5 - input_ts is mean reverting
            # hurst = 0.5 - input_ts is effectively random/geometric brownian motion
            # hurst > 0.5 - input_ts is trending
            tau = []
            lagvec = []  
            #  Step through the different lags  
            for lag in range(2, lags_to_test):  
                #  produce price difference with lag  
                pp = np.subtract(input_ts[lag:].values, input_ts[:-lag].values)  
                #  Write the different lags into a vector  
                lagvec.append(lag)  
                #  Calculate the variance of the differnce vector  
                tau.append(np.sqrt(np.std(pp)))  
            #  linear fit to double-log graph (gives power)  
            m = np.polyfit(np.log10(lagvec), np.log10(tau), 1)  
            # calculate hurst  
            hurst = m[0]*2
            print(hurst)
            return hurst  

        # dataframe for a historical instrument feature (ma_90 in this case). The index is the timestamps
        # of upto lookback data points. The columns of this dataframe are the stock symbols/instrumentIds.
        mom10Data = lookbackInstrumentFeatures.getFeatureDf('mom_10')
        mom30Data = lookbackInstrumentFeatures.getFeatureDf('mom_30')
        ma90Data = lookbackInstrumentFeatures.getFeatureDf('ma_90')
        
        # Here we are making predictions on the basis of Hurst exponent if enough data is available, otherwise
        # we simply get out of our position
        if len(ma90Data.index)>20:
            mom30 = mom30Data.iloc[-1]
            mom10 = mom10Data.iloc[-1]
            ma90 = ma90Data.iloc[-1]
            
            # Calculate Hurst Exponent
            hurst = ma90Data.apply(hurst_f, axis=0)
            # Go long if Hurst > 0.5 and both long term and short term momentum are positive
            predictions[(hurst > 0.5) & (mom30 > 0) & (mom10 > 0)] = 1 
            # Go short if Hurst > 0.5 and both long term and short term momentum are negative
            predictions[(hurst > 0.5) & (mom30 <= 0) & (mom10 <= 0)] = 0 
            
            # Get out of position if Hurst > 0.5 and long term momentum is positive while short term is negative
            predictions[(hurst > 0.5) & (mom30 > 0) & (mom10 <= 0)] = 0.5
            # Get out of position if Hurst > 0.5 and long term momentum is negative while short term is positive
            predictions[(hurst > 0.5) & (mom30 <= 0) & (mom10 > 0)] = 0.5
            
            # Get out of position if Hurst < 0.5
            predictions[hurst <= 0.5] = 0.5        
        else:
            # If no sufficient data then don't take any positions
            predictions.values[:] = 0.5
        return predictions

    def updateCount(self):
        self.count = self.count + 1
		
		
		
tf = MyTradingFunctions()
tsParams = MomentumTradingParams(tf)
tradingSystem = TradingSystem(tsParams)



results = tradingSystem.startTrading()