import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import pickle

import warnings
warnings.filterwarnings("ignore")

def read_data_and_preprocessing():
    df= pd.read_csv('data/data_mobil_gaikindo.csv', na_values=['-','0'])
    df= df.loc[:,:'ORIGIN COUNTRY']
    df.drop(['CC AND FUEL TYPE', 'ASSEMBLER', 'GEAR RATIO', 'PS / HP', 'WHEEL BASE', 'Unnamed: 17', 'CBU', 'ORIGIN COUNTRY','SPEED',], axis=1, inplace=True)

    df= copy_data_down(df, 'CAR TYPE')
    df= copy_data_down(df, 'BRAND')

    valid_model_type = ['TOYOTA VIOS', 'BMW 3', 'BMW 5', 'BMW 7', 'MERCEDES-BENZ PC', 'DAIHATSU Grand New  Xenia',
                        'DAIHATSU Luxio', 'DAIHATSU All New Terios', 'DAIHATSU Gran Max', 'DFSK GLORY',
                        'HONDA All New Jazz', 'HONDA New  Mobilio', 'HONDA HR-V', 'HONDA BR-V', 'HONDA All New CR-V',
                        'HONDA Brio', 'MITSUBISHI MOTORS', 'SUZUKI APV', 'SUZUKI All New Ertiga', 'SUZUKI XL07',
                        'TOYOTA YARIS', 'TOYOTA Sienta', 'TOYOTA AVANZA', 'TOYOTA RUSH', 'WULING Confero',
                        'WULING Cortez',
                        'WULING Formo', 'WULING Almaz', 'WULING Captiva', 'BMW X1', 'MINI Cooper', 'DFSK GLORY',
                        'HYUNDAI HIM', 'ISUZU Panther', 'TOYOTA INNOVA', 'TOYOTA FORTUNER', 'BMW X3', 'BMW X5',
                        'BMW X7',
                        'HINO FB', 'HINO WU', 'MITSUBISHI FUSO', 'HINO AK', 'HINO RK', 'HINO FC', 'HINO RN',
                        'MERCEDES-BENZ CV', 'ISUZU Traga', 'SUZUKI New Carry', 'DFSK DXK', 'HINO XZU', 'ISUZU NQR',
                        'ISUZU NMR', 'ISUZU NPS', 'ISUZU FRR', 'ISUZU NLR', 'TOYOTA DYNA', 'HINO FG', 'ISUZU FVR',
                        'ISUZU GVR', 'ISUZU FTR', 'HINO FL', 'HINO FM', 'ISUZU GVZ', 'ISUZU FVZ', 'ISUZU GXZ',
                        'ISUZU FVM',
                        'UD GKE', 'UD CDE', 'UD CWE', 'UD GWE', 'DAIHATSU New Ayla', 'DAIHATSU New Sigra', 'DATSUN GO',
                        'HONDA Brio', 'SUZUKI Karimun', 'TOYOTA AGYA', 'TOYOTA CALYA', 'TOYOTA TOWN / LITE', 'HINO SG']

    valid_model_type_by_brand = {}

    for i in valid_model_type:
        j = i.split(' ')
        brand = j[0].upper()
        type_model = (' '.join([str(i) for i in j[1:]])).upper().replace('  ', ' ')

        if (brand not in valid_model_type_by_brand):
            valid_model_type_by_brand[brand] = set([])

        valid_model_type_by_brand[brand].add(type_model)

    type_list = []
    for i in df.index:
        res_list = []
        car_type = df.loc[i, 'CAR TYPE'].replace('TYPE', '').strip()
        brand = df.loc[i, 'BRAND'].upper()
        model_type = df.loc[i, 'TYPE / MODEL'].upper().replace('  ', ' ')

        if (brand in valid_model_type_by_brand):
            for j in valid_model_type_by_brand[brand]:
                if (j in model_type):
                    res_list.append(j)

        else:
            for k in valid_model_type_by_brand.keys():
                if (k in brand):
                    brand = k
                    k.upper().replace('  ', ' ')
                    break

            if (brand in ['MERCEDES-BENZ', 'MITSUBISHI', 'HYUNDAI']):
                for l in valid_model_type_by_brand[brand]:
                    if (l in df.loc[i, 'BRAND']):
                        res_list.append(l)

            else:
                for m in valid_model_type_by_brand[brand]:
                    if (m in model_type):
                        res_list.append(m)

        if (len(res_list) > 1):
            if ('BMW' in model_type):
                model_type_temp = model_type.split(' ')
                if ('X' in model_type_temp[1]):
                    model_type2 = ''.join([i for i in res_list if 'X' in i])
                else:
                    model_type2 = model_type_temp[1][0]

            if ('HINO' in brand):
                for i in res_list:
                    if (i == model_type[:2] or i == model_type[:3]):
                        model_type2 = i
                        break

        if (len(res_list) > 1):
            type_list.append(f'{car_type}|||{brand} {model_type2}')
        else:
            type_list.append(f'{car_type}|||{brand} {res_list[0]}')

    df['TYPE_BRAND_MODEL'] = type_list
    df2 = df.copy()

    df2.drop(['CAR TYPE', 'BRAND', 'TYPE / MODEL'], axis=1, inplace=True)

    length_list = []
    width_list = []
    height_list = []

    for i in df2.index:
        dimension = df2.loc[i, 'DIMENSION P x L x T']
        if (dimension is not np.nan):
            dimension_list = dimension.upper().split('X')
            if (len(dimension_list) == 3):
                length_list.append(int(dimension_list[0].replace(',', '').replace('.', '').strip()))
                width_list.append(int(dimension_list[1].replace(',', '').replace('.', '').strip()))
                height_list.append(int(dimension_list[2].replace(',', '').replace('.', '').strip()))
            elif (len(dimension_list) == 2):
                length_list.append(int(dimension_list[0].replace(',', '').replace('.', '').strip()))
                width_list.append(int(dimension_list[1].replace(',', '').replace('.', '').strip()))
                height_list.append(np.nan)
            elif (len(dimension_list) == 1):
                length_list.append(dimension_list[0].replace(',', '').replace('.', '').strip())
                width_list.append(np.nan)
                height_list.append(np.nan)
        else:
            length_list.append(np.nan)
            width_list.append(np.nan)
            height_list.append(np.nan)

    df2['LENGTH'] = length_list
    df2['WIDTH'] = width_list
    df2['HEIGHT'] = height_list
    df2.drop('DIMENSION P x L x T', axis=1, inplace=True)

    for i in ['CC', 'GVW (Kg)', 'SEATER']:
        df2[i] = df2[i].map(validate_float).astype(float)

    df2['TRANS'] = df2['TRANS'].map(transmission_wrangling)
    df2.drop('WHEEL & TYRE SIZE', axis=1, inplace=True)

    df2['TRANS'] = df2['TRANS'].map(trans_change)
    df2['FUEL (G/D)'] = df2['FUEL (G/D)'].map(fuel_change)

    temp = df2['TYPE_BRAND_MODEL']
    df2.drop('TYPE_BRAND_MODEL', axis='columns', inplace=True)
    df2['TYPE_BRAND_MODEL'] = temp

    X = df2.drop('TYPE_BRAND_MODEL', axis='columns')
    y = df2['TYPE_BRAND_MODEL']

    df3 = pd.concat([pd.get_dummies(X), y], axis=1)

    return df, df2, df3

def building_model(df):
    X = df.drop('TYPE_BRAND_MODEL', axis='columns').values
    y = df['TYPE_BRAND_MODEL'].values

    xgb_model = XGBClassifier(random_state=0, max_delta_step=1)
    xgb_model.fit(X, y)

    return xgb_model

def copy_data_down(df, col):
    temp_data= ''

    for i in df.index:
        if(df.loc[i, col] is not np.nan):
            temp_data= df.loc[i, col]
        else:
            df.loc[i, col]= temp_data

    return df

def validate_float(x):
    if(x is not np.nan):
        if('+' in x):
            return int(x.split('+')[0])+int(x.split('+')[1])
        else:
            return x.replace(',','').replace('.','').strip()
    else:
        return np.nan

def transmission_wrangling(x):
    if (x is np.nan):
        return x

    if ('/' in x):
        if ('A' in x):
            return 'AT'
        else:
            return 'MT'
    else:
        return x

def trans_change(x):
    if x is np.nan:
        return x

    if x == 'MT':
        return 'Manual'
    elif x == 'AT':
        return 'Otomatis'
    elif x == 'CVT':
        return 'Transmisi CVT'

def fuel_change(x):
    if x is np.nan:
        return x

    if x == 'G':
        return 'Bensin'
    elif x == 'EV':
        return 'Listrik'
    elif x == 'D':
        return 'Diesel'

def validate_new_data(data, df2, df3):
    col_list = df2.columns[:-1]
    col_list2 = df3.columns[:-1]
    new_data = pd.DataFrame([data], columns=col_list)

    new_data = pd.get_dummies(new_data)
    missing_columns = set(col_list2) - set(new_data.columns)

    for i in missing_columns:
        new_data[i] = 0

    new_data = new_data[col_list2].values
    print(new_data)

    return new_data

def make_model_pickle():
    df, df2, df3 = read_data_and_preprocessing()
    xgb_model = building_model(df3)

    return xgb_model

def make_df2_pickle():
    df, df2, df3 = read_data_and_preprocessing()

    return df2

def make_df3_pickle():
    df, df2, df3 = read_data_and_preprocessing()

    return df3


def pred(new_data):
    try:
        df2 = pickle.load(open("df2.pkl", "rb"))
    except(OSError, IOError) as e:
        pickle.dump(make_df2_pickle(), open("df2.pkl", "wb"))
        df2 = pickle.load(open("df2.pkl", "rb"))

    try:
        df3 = pickle.load(open("df3.pkl", "rb"))
    except(OSError, IOError) as e:
        pickle.dump(make_df3_pickle(), open("df3.pkl", "wb"))
        df3 = pickle.load(open("df3.pkl", "rb"))

    try:
        xgb_model = pickle.load(open("model.pkl", "rb"))
    except(OSError, IOError) as e:
        pickle.dump(make_model_pickle(), open("model.pkl", "wb"))
        xgb_model = pickle.load(open("model.pkl", "rb"))

    data = validate_new_data(new_data, df2, df3)
    pred_y = xgb_model.predict(data)

    return pred_y
