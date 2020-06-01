import requests
import datetime
import numpy as np
import pandas as pd


today_dt = datetime.date.today()
today = today_dt.strftime('%Y-%m-%d')
OURWORLDINDATA = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
REPO_DATA = (
    'https://raw.githubusercontent.com/'
    'mauforonda/covid19-bolivia/master/data.json'
)


def fetch_from_ourworldindata(iso_code=None):
    response = requests.get(OURWORLDINDATA)

    with open('ourworldindata.csv', 'w+') as f:
        f.write(response.text)

    with open(f'data/ourworldindata_{today}.csv', 'w+') as f:
        f.write(response.text)

    df = pd.read_csv('ourworldindata.csv')

    return df


def fetch_from_covid19_bolivia_repo():
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
    if source == 'ourworldindata':
        filtered = fetch_from_ourworldindata()

    elif source == 'github':
        return fetch_from_covid19_bolivia_repo()
    elif filtered == 'boliviasegura':
        url = 'https://boliviasegura.agetic.gob.bo/wp-content/json/api.php'
        filtered = requests.get(url).json()

    filtered.to_csv(f'data/{source}.csv', index=False)
    filtered.to_csv(f'data/{source}_{today}.csv', index=False)
    filtered.sort_values(by='ds', inplace=True)
    return filtered


def get_population():
    population = pd.read_csv('data/population.csv')
    population = population[population['Year'] == '2019'].sort_values(['Population'], ascending=False)
    population = population[pd.notna(population['Code'])]
    return population


def get_full_data(source='ourworldindata', force=False):
    if force:
        df = fetch_from_ourworldindata()
    else:
        df = pd.read_csv(f'{source}.csv')

    df = df[df['iso_code'] != 'OWID_WRL']
    df = df[pd.notnull(df['iso_code'])]

    population = get_population()
    df = df.set_index('iso_code').join(population.set_index('Code')).reset_index()
    df.rename(columns={'index': 'iso_code'}, inplace=True)
    return df