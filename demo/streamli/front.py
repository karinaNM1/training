from datetime import datetime
import requests
import streamlit as st
import pandas as pd


# метод для вывода справки
def show_help():
    st.write("Добро пожаловать в справочное приложение!")
    st.write("Это справка предоставляет информацию о его функциональности.")
    st.write("С помощью данного виджета, вы можете просмотреть заражения covid-19 за определённый период "
             "в какой либо стране. Введите название страны на английском, и нажмите кнопку \"Отобразить\","
             " после данных действий загрузится светофор с отображением уровня опасности и график заражений за период. "
             "Так же вы можете указать период, просто выберите дату в полях."
             " ")
    st.write("Цвета светофора: зелёный - безопасно, желтый - средняя опасность, красный - очень опасно.")
    st.write("Данные предоставлены от 2020-01-01 до 2024-06-03.")


# метод нажания на кнопку "Отобразить"
def on_button_click():
    if st_country is not None:
        df = pd.DataFrame()
        lvl = '0'
        days = []
        mapping = {'Все': -1, 'Малый': 0, 'Средний': 1, 'Большой': 2}
        selected_value_numeric = mapping[st_hazard_lvl]

        # запись данный из полей ввода в data
        data = {
            "country_name": st_country,
            "from_date": st_from_date.strftime('%Y-%m-%d'),
            "before_date": st_before_date.strftime('%Y-%m-%d'),
            "hazard_lvl": selected_value_numeric
        }

        # обращение к апи и получение данных заражений
        url_get_calendar_information = "http://127.0.0.1:8000/get_calendar_information"
        response_calendar = requests.post(url_get_calendar_information, json=data)
        result_calendar = response_calendar.json()
        df = pd.DataFrame(result_calendar.get("calendar_information"))

        # обращение к апи и получение опасности заражения
        url_dangerous = "http://127.0.0.1:8000/predict_dangerous_country"
        response_dangerous = requests.post(url_dangerous, json=data)
        result_dangerous = response_dangerous.json()
        lvl = result_dangerous.get("danger_level")

        # обращение к апи и получение дат с определённым уровнем заражения
        if selected_value_numeric >= 0:
            url_days_dangerous = "http://127.0.0.1:8000/days_hazard_lvl"
            response_days_dangerous = requests.post(url_days_dangerous, json=data)
            result_days_dangerous = response_days_dangerous.json()
            days = (result_days_dangerous.get("days"))

        st.write("""
        #### Уровень опасности 
        """)

        # в зависимости от уровня опасности выводим нужную картинку
        path = ''
        if lvl == '1':
            path = 'draws/yellow.jpg'
        elif lvl == '0':
            path = 'draws/green.jpg'
        elif lvl == '2':
            path = 'draws/red.jpg'
        st.image(path, caption='Уровень опасности за период', use_column_width=False, width=200)

        st.markdown(f"""
            #### Заражения в {st_country} от {st_from_date} до {st_before_date}
        """)

        df = df[['date', 'new_cases']]
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)

        # Извлечение только даты из индекса
        df.index = df.index.date
        st.line_chart(df['new_cases'])

        # вывод дат если пользователь указывал уровень опасности
        if selected_value_numeric >= 0:
            st.markdown(f"""
                    #### Даты заражения уровня \"{st_hazard_lvl}\" в {st_country} от {st_from_date} до {st_before_date}
                """)
            if len(days) <= 0:
                st.write("Данных такого уровня опасности нет.")
            else:
                dates_text = '\n'.join(days)
                st.text_area("Даты:", value=dates_text, height=200)

    else:
        st.write("Введите название страны.")


# меню
st.sidebar.title('Меню')
menu_selection = st.sidebar.radio('Выберите раздел:', ['Виджет', 'Справка'])

if menu_selection == 'Виджет':
    st.title("Туристический виджет")

    from_date = datetime(2024, 3, 1)
    before_date = datetime(2024, 4, 30)

    # создание полей ввода
    st_country = st.text_input("Страна:", value='Morocco')
    st_from_date = st.date_input("Период от:", value=from_date)
    st_before_date = st.date_input("Период до:", value=before_date)
    st_hazard_lvl = st.selectbox(
        'Уровень опасности',
        ('Все', 'Малый', 'Средний', 'Большой')
    )

    # собитые нажатия на кнопку
    if st.button('Отобразить'):

        if pd.to_datetime(st_from_date) > pd.to_datetime(st_before_date):
            st.write('Первая дата должна быть меньше второй.')

        elif (pd.to_datetime(st_from_date) < pd.to_datetime('2020-01-01') or pd.to_datetime(st_from_date) > pd.to_datetime('2024-06-03')) \
                or (pd.to_datetime(st_before_date) < pd.to_datetime('2020-01-01') or pd.to_datetime(st_before_date) > pd.to_datetime('2024-06-03')):
            st.write('Данные предоставлены от 2020-01-01 до 2024-06-03')

        else:
            try:
                on_button_click()
            except:
                st.write('Ошибка')

elif menu_selection == 'Справка':
    st.title('Справка')
    show_help()


