# Домашняя работа 2 (5 баллов)
## Дедлайн
Работы принимаются до **21.11.2022 12:00 (понедельник)**.

## Требования
Работа должна быть выполнена на языке Python (версия не менее 3.6). Код должен быть написан аккуратно и читаемо, названия переменных должны отражать свою суть. Так же следует оставлять пояснения к своим подходам к решению.

Результат работы - ноутбук с вызываемым рабочим кодом и **комментариями-пояснениями**. Весь код должен быть закоммичен и создан Pull Request. В чате телеграма **нужно отметить @e_sheina и написать**, что Ваш PR готов к проверке и прислать ссылку на PR. Так же необходимо проставить 2 хеш-тега: #HW2 #Фамилия

## Отправка ДЗ
Репозиторий: https://github.com/SheinaEkaterina/Recsys-course-homework_2022

Правила именования PR:
- HW2: фамилия.первая_буква_имени (например, HW2: ivanov.s)

Правила именования ноутбуков: 
- В папке task_2_fm должен быть результирующий ноутбук, названный HW2.ipynb, и необходимые скрипты для решения.
- **Коммитить исходные данные не нужно!**

Пожалуйста, отнеситесь внимательно к отправке домашнего задания!

## Задание
Как и в первом домашнем задании, необходимо построить рекоммендательную систему, только не с помощью линейных моделей, а с применением FM.
- Постройте модель, которая предсказывает вероятность клика.
- Использовать oaid_hash нужно.
- Подберите гиперпараметр размерности для FM, сделав при этом валидацию. В качестве метрики считайте **log-loss** и **auc**.
- Свою итоговую модель примените к последнему дню датасета и вычислите log-loss и auc. Сравните результаты с результатами первой домашней работы.
- Для получения **максимального балла** необходимо реализовать FFM. Важно: правильно понять и реализовать указание филдов.

Данные лежат здесь: https://drive.google.com/drive/folders/1sxIGN-GUsJJcDF9WSg3Kzrgmzu1gp4PO

### Данные
Данные - это реальный лог реальных рекламных событий ad network PropellerAds, тот же, что и в первой домашней работе.

Датасет состоит из показов рекламы. 

Датасет состоит из показов рекламы. 

- date_time - время показа рекламы
- zone_id - id зоны, где зона - место на сайте для размещения рекламы
- banner_id - id баннера, где баннер - сама реклама
- campaign_clicks - общее количество показов данной кампании (которой соотвествует баннер) данному юзеру, произошедшие до текущего показа. Кампанию стоит понимать как что-то общее (рекламодатель/тематика/ и т. п.) для баннеров.
- os_id - id операционной системы
- country_id - id страны
- oaid_hash - хэш юзера
- banner_id0 - нулевой баннер в “стакане” баннеров
- banner_id1 - перый баннер в “стакане” баннеров
- rate0 - стоимость 1 клика установленная рекламодателем для banner_id0
- rate1 - стоимость 1 клика установленная рекламодателем для banner_id1
- g0 - стандартное отклонение предикта с banner_id0
- g1 - стандартное отклонение предикта с banner_id1
- coeff_sum0 - сумма коэффициентов для banner_id0
- coeff_sum1 - сумма коэффициентов для banner_id1
- impressions - был ли показ
- clicks - был ли клик

Для Домашней работы 2, колонки: banner_id0, banner_id1, rate0, rate1, g0, g1, coeff_sum0, coeff_sum1 использовать не нужно! Они будут использоваться в последующих Домашних работах.

## Вопросы
Все вопросы просьба задавать в чат, чтобы другие студенты могли найти ответы на схожие вопросы.
