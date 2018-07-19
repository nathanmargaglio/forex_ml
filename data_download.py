import requests, zipfile, io, os, time
import pandas as pd

def pickle_data(data_file):
    dff = pd.read_csv('{}'.format(data_file), header=None)
    dff = dff.rename(columns = {0: "pair", 1: "datetime", 2: "bid", 3: "ask"})

    dff.datetime = pd.to_datetime(dff.datetime)
    dff = dff.set_index('datetime')
    dff['price'] = dff[['bid', 'ask']].mean(axis=1)
    dff = dff.drop(columns=['bid', 'ask'])

    _open = dff.resample('1T').first().price
    _close = dff.resample('1T').last().price
    _high = dff.resample('1T').max().price
    _low = dff.resample('1T').min().price
    _vol = dff.resample('1T').count().price

    dff = dff.resample('1T').first()
    dff['open'] =_open
    dff['close'] = _close
    dff['high'] = _high
    dff['low'] = _low
    dff['volume'] = _vol
    dff = dff.drop(columns=['price'])
    
    dff = dff.fillna(method='ffill')

    dff.to_pickle(data_file.split('.csv')[0] + '.pickle')

pairs = ['AUDJPY', 'AUDNZD', 'AUDUSD', 'CADJPY', 'CHFJPY', 'EURGBP', 'EURJPY', 'EURUSD', 'GBPJPY', 'GBPUSD', 'NZDUSD', 'USDCAD']
years = ['2012', '2013', '2014', '2015', '2016', '2017']
months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE', 'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']

os.makedirs('data', exist_ok=True)
current_files = os.listdir('data')
processed = [f.split('.pickle')[0] for f in os.listdir('data')]

for year in years:
    for m, month in enumerate(months):
        for pair in pairs:
            print("{}: {} {} - {}".format(time.ctime(), month, year, pair))
            base_url = "http://truefx.com/dev/data/"
            url = base_url + "{}/{}-{}/{}-{}-{:02}.zip".format(year, month, year, pair, year, m+1)
            alt_url = base_url + "{}/{}-{:02}/{}-{}-{:02}.zip".format(year, year, m+1, pair, year, m+1)
            
            try:
                r = requests.get(url, stream=True)

                if not r.ok:
                    r = requests.get(alt_url, stream=True)

                if not r.ok:
                    print("Error: " + url)
                    continue

                z = zipfile.ZipFile(io.BytesIO(r.content))
                zip_filename = z.infolist()[0].filename
                data_file = 'data/' + zip_filename

                if zip_filename in current_files:
                    print(zip_filename + " already downloaded.")
                elif zip_filename.split('.csv')[0] + '.pickle' in current_files:
                    print(zip_filename.split('.csv')[0] + '.pickle' + " already generated.")
                    continue
                else:
                    print("Extracting...")
                    z.extractall('data')

                pickle_data(data_file)
                os.remove(data_file)
            except:
                print("Error at " + url)
