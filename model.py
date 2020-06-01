import requests
import datetime
import pandas as pd
import numpy as np
from fbprophet import Prophet


OURWORLDINDATA = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
REPO_DATA = 'https://raw.githubusercontent.com/mauforonda/covid19-bolivia/master/data.json'


def get_from_ourworldindata(iso_code=None):
    response = requests.get(OURWORLDINDATA)

    with open('owid-covid-data.csv', 'w+') as f:
        f.write(response.text)
    df = pd.read_csv('owid-covid-data.csv')
    if iso_code is not None:
        filtered = df[df['iso_code'] == iso_code]

    filtered = filtered[['date', 'total_cases']]
    filtered.columns = ['ds', 'y']
    filtered.reset_index(drop=True)

    return filtered


def get_from_covid19_bolivia_repo():
    response = requests.get(REPO_DATA)
    data = response.json()

    rows = []
    for item in data['confirmados']:
        row = {'fecha': item['fecha']}
        row.update(item['dep'])
        rows.append(row)

    cities = [
        'la_paz', 'cochabamba', 'santa_cruz', 'oruro', 'potosí', 'tarija',
        'chuquisaca', 'beni', 'pando'
    ]
    df = pd.DataFrame(rows)
    df['total'] = df[cities].sum(axis=1)

    filtered = df[(['fecha', 'total'] + cities)]
    filtered.columns = ['ds', 'y'] + cities
    return filtered


def get_data(source='ourworldindata'):
    today_dt = datetime.date.today()
    today = today_dt.strftime('%Y-%m-%d')

    if source == 'ourworldindata':
        filtered = get_from_ourworldindata()

    elif source == 'github':
        return get_from_covid19_bolivia_repo()
    elif filtered == 'boliviasegura':
        url = 'https://boliviasegura.agetic.gob.bo/wp-content/json/api.php'
        filtered = requests.get(url).json()

    filtered.to_csv(f'data/{source}.csv', index=False)
    filtered.to_csv(f'data/{source}_{today}.csv', index=False)
    filtered.sort_values(by='ds', inplace=True)
    return filtered



# Estimated total number of infected people for countries with similar
# population.
MIN_CAPACITY = 35000
MAX_CAPACITY = 75000

# source: https://www.ine.gob.bo/subtemas_cuadros/demografia_html/PC20106.htm
BOL = 11.63
CBBA = 2.03 / BOL
LP = 2.93 / BOL
SC = 3.37 / BOL
OR = 0.56 / BOL
BE = 0.48 / BOL

CAPACITY = {
    'y': (MIN_CAPACITY, MAX_CAPACITY),
    'cochabamba': (MIN_CAPACITY * CBBA, MAX_CAPACITY * CBBA),
    'la_paz': (MIN_CAPACITY * LP, MAX_CAPACITY * LP),
    'santa_cruz': (MIN_CAPACITY * SC, MAX_CAPACITY * SC),
    'oruro': (MIN_CAPACITY * OR, MAX_CAPACITY * OR),
    'beni': (MIN_CAPACITY * BE, MAX_CAPACITY * BE),
}


def get_forecast(filtered, capacity, periods=30, growth='logistic'):
    copied = filtered.copy()
    copied['cap'] = capacity
    m = Prophet(growth=growth)
    m.fit(copied)
    future = m.make_future_dataframe(periods=periods)
    future['cap'] = capacity
    forecast = m.predict(future)
    return m, forecast


def forecast(filtered, column='y', periods=30):
    copied = filtered[['ds', column]].copy()
    copied.columns = ['ds', 'y']
    full = copied.set_index('ds')

    # Min
    mlog_min, logistic_min = get_forecast(copied, CAPACITY[column][0], periods=periods)
    full = logistic_min[['ds', 'yhat']].set_index('ds').join(full)
    full.columns = ['y_optimistic', 'y']

    # Max
    mlog_max, logistic_max = get_forecast(copied, CAPACITY[column][1], periods=periods)
    full = logistic_max[['ds', 'yhat']].set_index('ds').join(full)
    full.columns = ['y_pessimistic', 'y_optimistic', 'y']

    full = full[['y', 'y_optimistic', 'y_pessimistic']]
    full = full.reset_index()

    full['y_predicted'] = full[['y_optimistic', 'y_pessimistic']].mean(axis=1)
    full = full[['ds', 'y', 'y_predicted', 'y_optimistic', 'y_pessimistic']]

    return full


def process_lower_upper(passed):
    df = passed.copy()
    upper = np.absolute(df['observed'] - df['pessimistic']).max()
    lower = np.absolute(df['observed'] - df['optimistic']).max()
    max_margin = max(upper, lower)

    df['optimistic'] = df['optimistic'] - max_margin
    df['pessimistic'] = df['pessimistic'] + max_margin

    df = df[['date', 'observed', 'predicted', 'optimistic', 'pessimistic']]
    df.columns = ['date', 'observed', 'predicted', 'optimistic', 'pessimistic']

    return df
