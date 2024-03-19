from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import pandas as pd
from arch import arch_model
from pydantic import BaseModel
from datetime import timedelta

# создания апи
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

np.set_printoptions(suppress=True)

# загрузка модели и датасета
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
df = pd.read_csv('result_1.csv')


# функция предсказывает уровень опасности
def check_dangerous(calendar_information):
    clf_predict_ = model.predict(calendar_information)
    calendar_information['clf_predict'] = clf_predict_
    return calendar_information


# функция добавляет столбец с предсказаниями
def preprocess_calendar_information(calendar_information):
    calendar_information2 = calendar_information.drop(columns=['location', 'date'])
    calendar_information2 = check_dangerous(calendar_information2)
    calendar_information['clf_predict'] = calendar_information2['clf_predict']
    return calendar_information


# функция передаёт данные по нужной стране и периоду
def get_calendar_information(country_name,  from_date, before_date):
    x = df[['location', 'date', 'total_cases', 'new_cases', 'new_cases_smoothed', 'total_deaths', 'new_deaths',
            'new_deaths_smoothed',
            'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million',
            'total_deaths_per_million',
            'new_deaths_per_million', 'new_deaths_smoothed_per_million', 'new_vaccinations_smoothed',
            'new_people_vaccinated_smoothed',
            'population_density', 'median_age', 'gdp_per_capita', 'extreme_poverty', 'female_smokers', 'male_smokers',
            'handwashing_facilities',
            'hospital_beds_per_thousand', 'human_development_index', 'population', 'RT']]
    x['date'] = pd.to_datetime(x['date'])
    if pd.to_datetime(from_date) >= pd.to_datetime('2024-02-04') or pd.to_datetime(before_date) >= pd.to_datetime(
            '2024-02-04'):
        x = month_prediction(country_name)
    calendar_information = x[(x['location'] == country_name) & (x['date'] >= pd.to_datetime(from_date)) &
                             (x['date'] <= pd.to_datetime(before_date))]
    return calendar_information


# функция возвращает предсказания уровня опасности за переод
def prediction(country_name, from_date, before_date):
    # получаем df с нужной страной и переодом
    calendar_information = get_calendar_information(country_name, from_date, before_date)
    # добавляем столбец с предсказанными значениями уровня
    calendar_information = preprocess_calendar_information(calendar_information)
    med = calendar_information[(calendar_information['date'] >= pd.to_datetime(from_date)) &
                               (calendar_information['date'] <= pd.to_datetime(before_date))]['clf_predict'].median()
    return str(round(med))


# функция возвращает даты с выбранным уровнем опасности
def get_days_hazard_lvl(country_name, from_date, before_date, hazard_lvl):
    # получаем df с нужной страной и переодом
    calendar_information = get_calendar_information(country_name, from_date, before_date)
    # добавляем столбец с предсказанными значениями уровня
    calendar_information = preprocess_calendar_information(calendar_information)
    # Фильтрация DataFrame по значению 'hazard_lvl' и выбор столбца 'date'
    filtered_dates = calendar_information[calendar_information['clf_predict'] == hazard_lvl]['date']
    # Преобразование дат в формат без времени
    dates_without_time = filtered_dates.dt.date.tolist()
    return dates_without_time


# предсказания данных на 3 месяца вперёд
def month_prediction(country_name):
    x = df[['location', 'date',
       'total_cases', 'new_cases', 'new_cases_smoothed',
       'total_deaths', 'new_deaths', 'new_deaths_smoothed',
        'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million',
        'total_deaths_per_million', 'new_deaths_per_million', 'new_deaths_smoothed_per_million',
        'new_vaccinations_smoothed', 'new_people_vaccinated_smoothed',
        'population_density', 'median_age', 'gdp_per_capita', 'extreme_poverty', 'female_smokers', 'male_smokers',
        'handwashing_facilities', 'hospital_beds_per_thousand', 'human_development_index', 'population', 'RT']]
    x['date'] = pd.to_datetime(x['date'])
    x.set_index('date', inplace=True)  # устанавливаем индеком дату для обучения модели

    # берём локацию которую выбрал пользователь
    df_loc = x[x['location'] == country_name]
    pred_size = 120

    # перебираем количество значений который нужно предсказать, обучаем модель и записываем предсказанные данные
    for i in range(pred_size):
        # предсказание новых случаев
        pred_cases = pred_garch(df_loc['new_cases_smoothed'])
        # предсказание новых смертей
        pred_deaths = pred_garch(df_loc['new_deaths_smoothed'])
        # записываем новую запись в df
        write_to_df(df_loc, pred_cases, pred_deaths)

    df_loc.reset_index(inplace=True)
    return df_loc


# функция записывает новую запись в df
def write_to_df(df, pred_c, pred_d):
    last_date = df.loc[df.index[-1]]
    new_case = round(np.sqrt(pred_c.variance.values[-1, :][0]))
    new_death = round(np.sqrt(pred_d.variance.values[-1, :][0]))
    total_cases = last_date['total_cases'] + new_case
    total_deaths = last_date['total_deaths'] + new_death
    population = last_date['population']

    # записываем новую запись в df, индекс это последняя дата плюс одна
    df.loc[pd.to_datetime(df.index[-1] + timedelta(days=1))] = [
        last_date['location'],
        total_cases, new_case, new_case,
        total_deaths, new_death, new_death,
        round(total_cases / (population / 1000000), 4), round(new_case / (population / 1000000), 4),
        round(new_case / (population / 1000000), 4),
        round(total_deaths / (population / 1000000), 4), round(new_death / (population / 1000000), 4),
        round(new_death / (population / 1000000), 4),
        last_date['new_vaccinations_smoothed'], last_date['new_people_vaccinated_smoothed'],
        last_date['population_density'], last_date['median_age'], last_date['gdp_per_capita'],
        last_date['extreme_poverty'], last_date['female_smokers'], last_date['male_smokers'],
        last_date['handwashing_facilities'], last_date['hospital_beds_per_thousand'],
        last_date['human_development_index'], population, last_date['RT']
    ]


# Предсказания моделью GARCH. Не возможно взять готовую и обученую модель, так как GARCH предсказывает один день и
#     должна заново обучаться с новыми данными.
def pred_garch(df_train):
    train = df_train
    model = arch_model(train, p=1, q=1,
                             mean='constant', vol='GARCH', dist='normal')
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    return pred


class Item(BaseModel):
    country_name: str
    from_date: str
    before_date: str
    hazard_lvl: int


@app.post("/get_calendar_information")
def calendar_information(item: Item):
    return {'calendar_information': get_calendar_information(item.country_name, item.from_date, item.before_date)}


@app.post("/predict_dangerous_country")
def predict(item: Item):
    return {'danger_level': prediction(item.country_name, item.from_date, item.before_date)}


@app.post("/days_hazard_lvl")
def days_hazard_lvl(item: Item):
    return {'days': get_days_hazard_lvl(item.country_name, item.from_date, item.before_date, item.hazard_lvl)}

