from src.base_config import Config


config = Config(
    data_path='data.csv',
    size=1000000,
    train_start_date='2021-09-26',
    train_end_date='2021-10-01',
    test_date='2021-10-02',
    no_matter_features={
        'oaid_hash',
        'banner_id0',
        'banner_id1',
        'rate0',
        'rate1',
        'g0',
        'g1',
        'coeff_sum0',
        'coeff_sum1'
    },
    features_to_generate=[
        {
            'features': ['hours'],
            'feature_name': 'per_hours',
        },
        {
            'features': ['country_id'],
            'feature_name': 'per_country',
        },
        {
            'features': ['zone_id'],
            'feature_name': 'per_zone',
        },
        {
            'features': ['banner_id'],
            'feature_name': 'per_banner',
        },
        {
            'features': ['os_id'],
            'feature_name': 'per_os',
        },
        {
            'features': ['hours', 'country_id'],
            'feature_name': 'per_hours_country',
        },
        {
            'features': ['os_id', 'country_id'],
            'feature_name': 'per_os_country',
        },
        {
            'features': ['weekday'],
            'feature_name': 'per_weekday',
        },
        {
            'features': ['weekday', 'hours'],
            'feature_name': 'per_weekday_hours',
        },
        {
            'features': ['weekday', 'country_id'],
            'feature_name': 'per_weekday_country',
        },
        {
            'features': ['weekday', 'country_id', 'hours'],
            'feature_name': 'per_weekday_country_hours',
        },
        {
            'features': ['hours', 'os_id', 'weekday'],
            'feature_name': 'per_weekday_os_hours',
        },
    ],
    features_to_train=[
        'clicks',
        'campaign_clicks',
        'per_hours',
        'per_country',
        'per_zone',
        'per_banner',
        'per_os',
        'per_hours_country',
        'per_os_country',
        'per_weekday',
        'per_weekday_hours',
        'per_weekday_country',
        'per_weekday_country_hours',
        'per_weekday_os_hours',
    ],
    random_state=42,
)
