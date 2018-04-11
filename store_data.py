import pandas as pd
from datetime import datetime
import math

CSV_COLUMN_NAMES = ['date_', 'store',
'department','item', 'unit_price', 'quantity','on_promotion','promotion_type']

PATH1 = './hackathon_dataset_2009.dat'
PATH2 = './hackathon_dataset_2010.dat'
PATH3 = './hackathon_dataset_2011.dat'
RESULT_PATH = './hackathon_result_copy.dat'
WRITE_PATH = './hackathon_result.dat'

def read_data():
    year1 = pd.read_csv(PATH1)
    year2 = pd.read_csv(PATH2)
    year3 = pd.read_csv(PATH3)
    join = pd.concat([year1,year2,year3])
    return join

def to_time(m):
    month = m//100
    day = m - month*100
    if month == 1: return day
    elif month == 2: return 31 + day
    else: return 59 + (month - 2)//2 + (month - 3)*30 + day

def to_cos_day(monthday):
    month = monthday//100
    day = monthday - month*100
    if month in (1,5,7,8,10,12):
        divider = 31
    elif month in (2,4,6,9,11):
        divider = 31
    else: divider = 28
    return math.cos(2*math.pi*day/divider)

def to_sin_day(monthday):
    month = monthday//100
    day = monthday - month*100
    if month in (1,5,7,8,10,12):
        divider = 31
    elif month in (2,4,6,9,11):
        divider = 31
    else: divider = 28
    return math.sin(2*math.pi*day/divider)

def to_cos_time(time):
    return math.cos(2*math.pi*time/365)

def to_sin_time(time):
    return math.sin(2*math.pi*time/365)

#Holiday criteria variables
christmas = [1125,1231]
halloween = [931,1107]
valentine = [114,221]
thanksgiving = [1022,1129]
st_patricks = [217,324]
holiday_time_periods = [christmas,halloween,valentine,thanksgiving,st_patricks]

for holiday in holiday_time_periods:
    for i in range(len(holiday)):
        holiday[i] = to_time(holiday[i])

def christmas_range(time):
    return int(christmas[0] <= time <= christmas[1])
def halloween_range(time):
    return int(halloween[0] <= time <= halloween[1])
def valentine_range(time):
    return int(valentine[0] <= time <= valentine[1])
def thanksgiving_range(time):
    return int(thanksgiving[0] <= time <= thanksgiving[1])
def st_patricks_range(time):
    return int(st_patricks[0] <= time <= st_patricks[1])
    
def load_data(test_percent,test_selection):
    data = read_data()
    unique_items = data['item'].unique()
    unique_stores = data['store'].unique()
    data.pop('on_promotion')
    data.department = data.department-1
    data['year'] = data.date_ // 10000
    data['month'] = (data.date_ - data.year * 10000)//100
    data['day'] = (data.date_ - data.year*10000 - data.month*100)
    monthday = data.month*100 + data.day
    data['time'] = monthday.apply(to_time)
    data['sin_time'] = data.time.apply(to_sin_time)
    data['cos_time'] = data.time.apply(to_cos_time)
    data['sin_day'] = monthday.apply(to_sin_day)
    data['cos_day'] = monthday.apply(to_cos_day)
    data.year = data.year - 2009
    data['total_time'] = data.year + data.time/365
        
    data['christmas'] = data.time.apply(christmas_range)
    data['halloween'] = data.time.apply(halloween_range)
    data['valentine'] = data.time.apply(valentine_range)
    data['thanksgiving'] = data.time.apply(thanksgiving_range)
    data['st_patricks'] = data.time.apply(st_patricks_range)
    
    if test_selection==0:
        #use test_percentage as the percentage of latest data used for testing
        num_test = math.ceil(data.shape[0]*(test_percent/100.0))
        total = data.shape[0]
        train = data[0:total-num_test]
        test = data[total-num_test:]
    else:
        #only use jan of 2012 for testing, everything else is training
        test = data.loc[(data['year'] == 2) & (data['month'] == 1)]
        train = data.loc[(data['year'] != 2) | (data['month'] != 1)]
    train_y = train.pop('quantity')
    train_x = train
    test_y = test.pop('quantity')
    test_x = test
    return (train_x,train_y),(test_x,test_y),(unique_items,unique_stores)

def save_predictions(m):
    #pass it a list of predictions. Converts it to series and adds it to hackathon_result.dat dataframe.
    #writes new dataframe to hackathon_result.dat
    #Just to be safe, we keep an unaltered copy of hackathon_result.dat called hackathon_result_copy.dat
    predictions = pd.Series(m)
    result = pd.read_csv(RESULT_PATH)
    result.quantity = predictions
    result.to_csv(WRITE_PATH)