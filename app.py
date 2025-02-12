import torch
import tsfel
import warnings
import pandas as pd
import polars as pl
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassAccuracy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score, classification_report, precision_score, recall_score, \
    f1_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Ignorar warnings específicos
warnings.filterwarnings('ignore', message='Precision loss occurred in moment calculation')

bokken_data = 'bokken_data'
shikko_data = 'shikko_data'

bokken_features_list = [
    'tsfel.mean'
    , 'tsfel.standard_deviation'
    , 'tsfel.variance'
    , 'tsfel.kurtosis'
    , 'tsfel.skewness'
    , 'tsfel.mean_abs_deviation'
    , 'tsfel.number_peaks'
    , 'tsfel.autocorrelation'
    , 'tsfel.zero_cross'
    , 'tsfel.mean_abs_diff'
    , 'tsfel.energy'
    , 'tsfel.spectral_entropy'
    , 'tsfel.spectral_centroid'
    , 'tsfel.spectral_spread'
    , 'tsfel.median_frequency'
    , 'tsfel.max_frequency'
    , 'tsfel.power_bandwidth'
]
shikko_features_list = [
    'tsfel.mean_abs_deviation'
    , 'tsfel.rms'
    , 'tsfel.calc_var'
    , 'tsfel.kurtosis'
    , 'tsfel.autocorr'
    , 'tsfel.zero_cross'
    , 'tsfel.median_abs_diff'
    , 'tsfel.pk_pk_distance'
    , 'tsfel.spectral_centroid'
    , 'tsfel.spectral_entropy'
    , 'tsfel.median_frequency'
    , 'tsfel.spectral_slope'
    , 'tsfel.wavelet_energy'
]

bokken_columns_to_process_list = [
    'ACCX_BOKKEN'
    , 'ACCY_BOKKEN'
    , 'ACCZ_BOKKEN'
    , 'MOD_ACC'
]
shikko_columns_to_process_list = [
    'Acc_X'
    , 'Acc_Y'
    , 'Acc_Z'
    , 'Gyr_X'
    , 'Gyr_Y'
    , 'Gyr_Z'
    , 'Mag_X'
    , 'Mag_Y'
    , 'Mag_Z'
    , 'Acc_X_sphe'
    , 'Acc_Y_sphe'
    , 'Acc_Z_sphe'
    , 'Gyr_X_sphe'
    , 'Gyr_Y_sphe'
    , 'Gyr_Z_sphe'
    , 'Mag_X_sphe'
    , 'Mag_Y_sphe'
    , 'Mag_Z_sphe'
    , 'Acc_X_cyli'
    , 'Acc_Y_cyli'
    , 'Acc_Z_cyli'
    , 'Gyr_X_cyli'
    , 'Gyr_Y_cyli'
    , 'Gyr_Z_cyli'
    , 'Mag_X_cyli'
    , 'Mag_Y_cyli'
    , 'Mag_Z_cyli'
    , 'MOD_ACC'
    , 'MOD_GYR'
    , 'FORCE'
    , 'Q1_madgwick_cart'
    , 'Q2_madgwick_cart'
    , 'Q3_madgwick_cart'
    , 'Q4_madgwick_cart'
    , 'Q1_madgwick_sphe'
    , 'Q2_madgwick_sphe'
    , 'Q3_madgwick_sphe'
    , 'Q4_madgwick_sphe'
    , 'Q1_madgwick_cyli'
    , 'Q2_madgwick_cyli'
    , 'Q3_madgwick_cyli'
    , 'Q4_madgwick_cyli'
]

def data_analysis_bokken():
    schema = {
        'Time': pl.Float64,       # Marca de tiempo
        'ACCX_BOKKEN': pl.Int64,  # Aceleración en X
        'ACCY_BOKKEN': pl.Int64,  # Aceleración en Y
        'ACCZ_BOKKEN': pl.Int64,  # Aceleración en Z
        'Name': pl.Utf8,          # Nombre del practicante
        'Type': pl.Utf8,          # Tipo de movimiento
        'Movement': pl.Utf8,      # Movimiento
        'Sample': pl.Int64,       # Muestra
        'MOD_ACC': pl.Float64,    # Magnitud de la aceleración
        'Gender': pl.Utf8,        # Género
        'Height': pl.Int64,       # Altura en cm
        'Weight': pl.Int64,       # Peso en kg
        'Age': pl.Int64,          # Edad
        'Experience': pl.Float64, # Experiencia
        'Kyu-Dan': pl.Int64,      # Rango de Kyu-Dan
        'Forearm': pl.Int64,      # Longitud del antebrazo
        'Arm': pl.Int64,          # Longitud del brazo
        'Other': pl.Utf8,         # Otro (arte marcial o información adicional)
        'BMI': pl.Float64         # Índice de masa corpora
    }

    # Cargar los datos
    data = pl.read_csv('bokken_data.csv', separator=',', schema=schema)

    # Seleccionar columnas para análisis y eliminar duplicados
    analysis = data.select(['Name', 'Height', 'Weight', 'Age', 'Forearm', 'Arm', 'BMI', 'Kyu-Dan']).unique()

    # Convertir a pandas para cálculo de correlación y visualización
    analysis_pandas = analysis.to_pandas()

    # Matriz de correlación para identificar relaciones entre las características y Kyu-Dan
    correlation_matrix = analysis_pandas[['Height', 'Weight', 'Age', 'Forearm', 'Arm', 'BMI', 'Kyu-Dan']].corr()

    # Graficar un heatmap para visualizar la correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix (Features vs Kyu-Dan)")
    plt.show()

    # Graficar distribuciones y relaciones entre características y Kyu-Dan
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes      = axes.flatten()
    features  = ['Height', 'Weight', 'Age', 'Forearm', 'Arm', 'BMI']

    for i, feature in enumerate(features):
        sns.scatterplot(x=analysis_pandas[feature], y=analysis_pandas['Kyu-Dan'], ax=axes[i], alpha=0.7)
        axes[i].set_title(f"{feature} vs Kyu-Dan")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Kyu-Dan")

    plt.tight_layout()
    plt.show()

    # Estadísticas descriptivas de las características seleccionadas
    descriptive_stats = analysis.select(['Height', 'Weight', 'Age', 'Forearm', 'Arm', 'BMI']).describe()

    # Guardar estadísticas descriptivas en un archivo CSV para revisión
    descriptive_stats_pandas = descriptive_stats.to_pandas().round(2)
    descriptive_stats_pandas.to_csv("descriptive_statistics_polars.csv", index=True)

    print("\nEstadisticas descriptivas:")
    print(descriptive_stats_pandas)

    return data
def data_analysis_shikko():
    schema = {
        'id': pl.Int64,                 # Identificador único
        'Name': pl.Utf8,                # Nombre del practicante
        'Gender': pl.Utf8,              # Género
        'Height': pl.Int64,             # Altura en cm
        'Weight': pl.Int64,             # Peso en kg
        'Age': pl.Int64,                # Edad en años
        'Experience': pl.Float64,       # Experiencia en años
        'Kyu-Dan': pl.Int64,            # Nivel de Kyu o Dan
        'Other': pl.Utf8,               # Otra práctica
        'BMI': pl.Float64,              # Índice de Masa Corporal
        'Time': pl.Utf8,                # Marca de tiempo
        'Acc_X': pl.Float64,            # Aceleración en X
        'Acc_Y': pl.Float64,            # Aceleración en Y
        'Acc_Z': pl.Float64,            # Aceleración en Z
        'Gyr_X': pl.Float64,            # Giroscopio en X
        'Gyr_Y': pl.Float64,            # Giroscopio en Y
        'Gyr_Z': pl.Float64,            # Giroscopio en Z
        'Mag_X': pl.Int64,              # Magnetómetro en X
        'Mag_Y': pl.Int64,              # Magnetómetro en Y
        'Mag_Z': pl.Int64,              # Magnetómetro en Z
        'Acc_X_sphe': pl.Float64,       # Aceleración esférica en X
        'Acc_Y_sphe': pl.Float64,       # Aceleración esférica en Y
        'Acc_Z_sphe': pl.Float64,       # Aceleración esférica en Z
        'Gyr_X_sphe': pl.Float64,       # Giroscopio esférico en X
        'Gyr_Y_sphe': pl.Float64,       # Giroscopio esférico en Y
        'Gyr_Z_sphe': pl.Float64,       # Giroscopio esférico en Z
        'Mag_X_sphe': pl.Float64,       # Magnetómetro esférico en X
        'Mag_Y_sphe': pl.Float64,       # Magnetómetro esférico en Y
        'Mag_Z_sphe': pl.Float64,       # Magnetómetro esférico en Z
        'Acc_X_cyli': pl.Float64,       # Aceleración cilíndrica en X
        'Acc_Y_cyli': pl.Float64,       # Aceleración cilíndrica en Y
        'Acc_Z_cyli': pl.Float64,       # Aceleración cilíndrica en Z
        'Gyr_X_cyli': pl.Float64,       # Giroscopio cilíndrico en X
        'Gyr_Y_cyli': pl.Float64,       # Giroscopio cilíndrico en Y
        'Gyr_Z_cyli': pl.Float64,       # Giroscopio cilíndrico en Z
        'Mag_X_cyli': pl.Float64,       # Magnetómetro cilíndrico en X
        'Mag_Y_cyli': pl.Float64,       # Magnetómetro cilíndrico en Y
        'Mag_Z_cyli': pl.Int64,         # Magnetómetro cilíndrico en Z
        'Type': pl.Utf8,                # Tipo de movimiento
        'Movement': pl.Utf8,            # Movimiento
        'Sample': pl.Int64,             # Número de muestra
        'MOD_ACC': pl.Float64,          # Módulo de la aceleración
        'MOD_GYR': pl.Float64,          # Módulo del giroscopio
        'FORCE': pl.Float64,            # Fuerza calculada
        'ARM_RATIO': pl.Float64,        # Proporción del brazo
        'Q1_default_cart': pl.Float64,  # Cuaternión 1 en sistema cartesiano (por defecto)
        'Q2_default_cart': pl.Float64,  # Cuaternión 2 en sistema cartesiano (por defecto)
        'Q3_default_cart': pl.Float64,  # Cuaternión 3 en sistema cartesiano (por defecto)
        'Q4_default_cart': pl.Float64,  # Cuaternión 4 en sistema cartesiano (por defecto)
        'Q1_madgwick_cart': pl.Float64, # Cuaternión 1 (Madgwick) en sistema cartesiano
        'Q2_madgwick_cart': pl.Float64, # Cuaternión 2 (Madgwick) en sistema cartesiano
        'Q3_madgwick_cart': pl.Float64, # Cuaternión 3 (Madgwick) en sistema cartesiano
        'Q4_madgwick_cart': pl.Float64, # Cuaternión 4 (Madgwick) en sistema cartesiano
        'Q1_mahony_cart': pl.Float64,   # Cuaternión 1 (Mahony) en sistema cartesiano
        'Q2_mahony_cart': pl.Float64,   # Cuaternión 2 (Mahony) en sistema cartesiano
        'Q3_mahony_cart': pl.Float64,   # Cuaternión 3 (Mahony) en sistema cartesiano
        'Q4_mahony_cart': pl.Float64,   # Cuaternión 4 (Mahony) en sistema cartesiano
        'Q1_kalman_cart': pl.Float64,   # Cuaternión 1 (Kalman) en sistema cartesiano
        'Q2_kalman_cart': pl.Float64,   # Cuaternión 2 (Kalman) en sistema cartesiano
        'Q3_kalman_cart': pl.Float64,   # Cuaternión 3 (Kalman) en sistema cartesiano
        'Q4_kalman_cart': pl.Float64,   # Cuaternión 4 (Kalman) en sistema cartesiano
        'Q1_default_sphe': pl.Float64,  # Cuaternión 1 (por defecto) en sistema esférico
        'Q2_default_sphe': pl.Float64,  # Cuaternión 2 (por defecto) en sistema esférico
        'Q3_default_sphe': pl.Float64,  # Cuaternión 3 (por defecto) en sistema esférico
        'Q4_default_sphe': pl.Float64,  # Cuaternión 4 (por defecto) en sistema esférico
        'Q1_madgwick_sphe': pl.Float64, # Cuaternión 1 (Madgwick) en sistema esférico
        'Q2_madgwick_sphe': pl.Float64, # Cuaternión 2 (Madgwick) en sistema esférico
        'Q3_madgwick_sphe': pl.Float64, # Cuaternión 3 (Madgwick) en sistema esférico
        'Q4_madgwick_sphe': pl.Float64, # Cuaternión 4 (Madgwick) en sistema esférico
        'Q1_mahony_sphe': pl.Float64,   # Cuaternión 1 (Mahony) en sistema esférico
        'Q2_mahony_sphe': pl.Float64,   # Cuaternión 2 (Mahony) en sistema esférico
        'Q3_mahony_sphe': pl.Float64,   # Cuaternión 3 (Mahony) en sistema esférico
        'Q4_mahony_sphe': pl.Float64,   # Cuaternión 4 (Mahony) en sistema esférico
        'Q1_kalman_sphe': pl.Float64,   # Cuaternión 1 (Kalman) en sistema esférico
        'Q2_kalman_sphe': pl.Float64,   # Cuaternión 2 (Kalman) en sistema esférico
        'Q3_kalman_sphe': pl.Float64,   # Cuaternión 3 (Kalman) en sistema esférico
        'Q4_kalman_sphe': pl.Float64,   # Cuaternión 4 (Kalman) en sistema esférico
        'Q1_default_cyli': pl.Float64,  # Cuaternión 1 (por defecto) en sistema cilíndrico
        'Q2_default_cyli': pl.Float64,  # Cuaternión 2 (por defecto) en sistema cilíndrico
        'Q3_default_cyli': pl.Float64,  # Cuaternión 3 (por defecto) en sistema cilíndrico
        'Q4_default_cyli': pl.Float64,  # Cuaternión 4 (por defecto) en sistema cilíndrico
        'Q1_madgwick_cyli': pl.Float64, # Cuaternión 1 (Madgwick) en sistema cilíndrico
        'Q2_madgwick_cyli': pl.Float64, # Cuaternión 2 (Madgwick) en sistema cilíndrico
        'Q3_madgwick_cyli': pl.Float64, # Cuaternión 3 (Madgwick) en sistema cilíndrico
        'Q4_madgwick_cyli': pl.Float64, # Cuaternión 4 (Madgwick) en sistema cilíndrico
        'Q1_mahony_cyli': pl.Float64,   # Cuaternión 1 (Mahony) en sistema cilíndrico
        'Q2_mahony_cyli': pl.Float64,   # Cuaternión 2 (Mahony) en sistema cilíndrico
        'Q3_mahony_cyli': pl.Float64,   # Cuaternión 3 (Mahony) en sistema cilíndrico
        'Q4_mahony_cyli': pl.Float64,   # Cuaternión 4 (Mahony) en sistema cilíndrico
        'Q1_kalman_cyli': pl.Float64,   # Cuaternión 1 (Kalman) en sistema cilíndrico
        'Q2_kalman_cyli': pl.Float64,   # Cuaternión 2 (Kalman) en sistema cilíndrico
        'Q3_kalman_cyli': pl.Float64,   # Cuaternión 3 (Kalman) en sistema cilíndrico
        'Q4_kalman_cyli': pl.Float64    # Cuaternión 4 (Kalman) en sistema cilíndrico
    }

    # Cargar los datos
    data = pl.read_csv('shikko_data.csv', separator=',', schema=schema)

    # Seleccionar columnas para análisis y eliminar duplicados
    analysis = data.select(['Name', 'Height', 'Weight', 'Age', 'Kyu-Dan', 'BMI']).unique()

    # Convert to pandas for correlation matrix and visualization
    analysis_pandas = analysis.to_pandas()

    # Calcular matriz de correlación
    correlation_matrix = analysis_pandas[['Height', 'Weight', 'Age', 'Kyu-Dan', 'BMI']].corr()

    # Graficar matriz de correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    plt.title("Matriz de Correlación (Características vs Kyu-Dan)")
    plt.show()

    # Graficar dispersión de cada característica contra Kyu-Dan
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes      = axes.flatten()
    features  = ['Height', 'Weight', 'Age', 'BMI']

    for i, feature in enumerate(features):
        sns.scatterplot(x=analysis_pandas[feature], y=analysis_pandas['Kyu-Dan'], ax=axes[i], alpha=0.7)
        axes[i].set_title(f"{feature} vs Kyu-Dan")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("Kyu-Dan")

    plt.tight_layout()
    plt.show()

    # Estadísticas descriptivas usando Polars
    descriptive_stats = analysis.describe().to_pandas().round(2)

    # Mostrar estadísticas descriptivas
    print("\nEstadísticas Descriptivas:")
    print(descriptive_stats)

    return data
def clean_data_bokken():

    data = data_analysis_bokken()

    # Seleccionar solo las columnas necesarias
    data = data.select(
        [col for col in data.columns if not any(
            keyword in col for keyword in [
                'id'
                ,'Weight'
                ,'Forearm'
                ,'Height'
                ,'Sample'
                ,'Other'
                ,'Movement'
                ,'Age'
            ]
        )]
    )

    scaler = StandardScaler()

    # Binarizar Type
    data = data.with_columns(
        pl.when(pl.col('Type') == 'Going').then(0).otherwise(1).alias('Type')
    )

    # Normalizar ['BMI', 'Arm']
    data_to_normalize = data.select(['BMI', 'Arm']).to_pandas()
    normalized_data   = scaler.fit_transform(data_to_normalize)
    data = data.with_columns(
        [pl.Series(['BMI', 'Arm'][i], normalized_data[:, i]) for i in range(len(['BMI', 'Arm']))]
    )

    # Contar los Time por 'Name' y asignar el conteo a cada fila correspondiente
    counts = data.group_by('Name').agg(pl.count('Name').alias('Time'))
    data   = data.drop('Time').join(counts, on='Name')

    # Binarizar Gender
    data = data.with_columns(
        pl.when(pl.col('Gender') == 'FEMALE').then(0).otherwise(1).alias('Gender')
    )

    # Generar datos fijos
    fixed_values = data.select('Name', 'Time', 'Gender', 'BMI', 'Kyu-Dan', 'Experience', 'Arm').unique()

    fixed_values.write_csv(f'bokken_data_fixed_values.csv', separator='|')
    data.write_csv(f'bokken_data_clean.csv', separator='|')
def clean_data_shikko():

    data = data_analysis_shikko()

    # Seleccionar solo las columnas necesarias
    data = data.select(
        [col for col in data.columns if not any(
            keyword in col for keyword in [
                'id'
                ,'Experience'
                ,'Type'
                ,'Height'
                ,'Weight'
                ,'Sample'
                ,'Other'
                ,'ARM_RATIO'
                ,'Movement'
                ,'default'
                ,'mahony'
                ,'kalman'
                ,'BMI'
            ]
        )]
    )

    scaler = StandardScaler()

    # Normalizar ['Age']
    data_to_normalize = data.select(['Age']).to_pandas()
    normalized_data   = scaler.fit_transform(data_to_normalize)
    data = data.with_columns(
        [pl.Series(['Age'][i], normalized_data[:, i]) for i in range(len(['Age']))]
    )

    # Segmentar las series en mitad y mitad creando Type
    data = (
        data.sort(['Name', 'Time']) # Asegurar el orden por Name y Time
        .group_by('Name', maintain_order=True)
        .map_groups(
            lambda group: group.with_columns(
                pl.Series('Type', [0] * (len(group) // 2) + [1] * (len(group) - len(group) // 2))
            )
        )
    )

    # Contar los Time por 'Name' y asignar el conteo a cada fila correspondiente
    counts = data.group_by('Name').agg(pl.count('Name').alias('Time'))
    data   = data.drop('Time').join(counts, on='Name')

    # Binarizar Gender
    data = data.with_columns(
        pl.when(pl.col('Gender') == 'FEMALE').then(0).otherwise(1).alias('Gender')
    )

    # Generar datos fijos
    fixed_values = data.select('Name', 'Time', 'Gender', 'Age', 'Kyu-Dan').unique()

    fixed_values.write_csv(f'shikko_data_fixed_values.csv', separator='|')
    data.write_csv(f'shikko_data_clean.csv', separator='|')
def create_feature_dataset(dataset, columns_to_process_list, features_list, window_size=40, overlap=0.5):
    # Cargar los datos
    data = pl.read_csv(f'{dataset}_clean.csv', separator='|')

    # Configuración de TSFEL (extrae solo las características deseadas)
    cfg = tsfel.get_features_by_domain()
    filtered_cfg = {domain: {k: v for k, v in cfg[domain].items() if v['function'] in features_list} for domain in cfg}

    # Lista para almacenar las características por grupo y segmento
    all_features_list = []

    # Agrupar por 'Name' y 'Type'
    grouped_data = data.group_by(['Name', 'Type'])

    count = 0

    # Procesar cada grupo (cada persona y tipo de movimiento)
    for group_key, group_df in grouped_data:
        # Ordenar por tiempo
        group_df = group_df.sort('Time')

        print('Name:', group_key, count)

        # Convertir a numpy para procesamiento
        signals_dict = {col: group_df[col].to_numpy() for col in columns_to_process_list}

        # Definir los índices de las ventanas deslizantes
        step_size   = int(window_size * (1 - overlap))  # Tamaño de paso basado en el solapamiento
        num_samples = len(group_df)

        for start in range(0, num_samples - window_size + 1, step_size):
            end = start + window_size

            # Crear un diccionario para almacenar características de la ventana
            window_features_dict = {}

            # Extraer características para cada serie temporal en la ventana
            for col, signal in signals_dict.items():
                segment = signal[start:end]
                features = tsfel.time_series_features_extractor(filtered_cfg, segment, fs=100, verbose=0)

                # Agregar las características al diccionario con prefijo de la columna
                for feature_col in features.columns:
                    window_features_dict[f'{col}_{feature_col}'] = features[feature_col].values[0]  # TSFEL devuelve un DF

            # Agregar identificadores
            window_features_dict['Name'] = group_key[0]
            window_features_dict['Type'] = group_key[1]
            window_features_dict['Segment_Start'] = start
            window_features_dict['Segment_End']   = end

            # Convertir a DataFrame de Polars y agregar a la lista
            all_features_list.append(pl.DataFrame([window_features_dict]))

        count += 1

    # Unir todas las características en un solo DataFrame
    features_df = pl.concat(all_features_list)

    # Cargar valores fijos y unir por 'Name'
    fixed_values = pl.read_csv(f'{dataset}_fixed_values.csv', separator='|')
    features_df = features_df.join(fixed_values, on='Name', how='left')

    # Guardar en CSV
    features_df.sort('Name').write_csv(f'{dataset}_features.csv', separator='|')

    print(f"Se generó el archivo {dataset}_features.csv con características por segmento.")
def create_selected_features_dataset(dataset):
    data = pl.read_csv(f'{dataset}_features.csv', separator='|')

    columns_without_nulls = [col for col in data.columns if data[col].is_null().sum() == 0 and (data[col].is_nan().sum() if data[col].dtype.is_float() else 0) == 0]
    data = data.select(columns_without_nulls)

    # Separar las características (X) y la etiqueta (y)
    X = data.select([col for col in data.columns if col not in ['Kyu-Dan', 'Time', 'Name', 'Type']])
    y = data['Kyu-Dan']

    # Convertir 'X' e 'y' a NumPy para usar con Scikit-Learn
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    # Escalar las características
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)

    # Ajustar el modelo LASSO
    alpha = 0.01 if dataset == 'bokken_data' else 0.2
    lasso = Lasso(alpha=alpha)  # Ajusta alpha según la regularización deseada
    lasso.fit(X_scaled, y_np)

    # Obtener las características seleccionadas
    selected_features = [col for col, coef in zip(X.columns, lasso.coef_) if coef != 0]
    print(f'Características seleccionadas [{len(selected_features)}]:')
    print(selected_features)

    # Crear un nuevo DataFrame con las características seleccionadas
    selected_data = data.select(['Name', 'Type', 'Kyu-Dan', *selected_features])

    # Crear una nueva columna categórica para Kyu-Dan
    selected_data = selected_data.with_columns(
        pl.when(pl.col('Kyu-Dan') >= 4).then(pl.lit(0))
        .when((pl.col('Kyu-Dan') >= -2) & (pl.col('Kyu-Dan') < 4)).then(pl.lit(1))
        .otherwise(pl.lit(2))
        .alias('Kyu-Dan')
    )

    # Guardar las características seleccionadas en un archivo CSV
    selected_data.write_csv(f'{dataset}_selected_features.csv', separator='|')
def prepare_data(dataset):
    if dataset == 'bokken_data':
        clean_data_bokken()
        column_list = bokken_columns_to_process_list
        features    = bokken_features_list
    else:
        clean_data_shikko()
        column_list = shikko_columns_to_process_list
        features    = shikko_features_list

    create_feature_dataset(dataset, column_list, features)
    create_selected_features_dataset(dataset)

# Models
# GPR
def apply_gpr(dataset):
    # Cargar el datasetset
    data = pd.read_csv(f'{dataset}_selected_features.csv', sep='|')

    # Inicializar StandardScaler
    scaler = StandardScaler()

    # DataFrame para guardar predicciones
    all_predictions = []

    # Parámetros óptimos para cada movement_type
    optimal_params = {
        0: {'alpha': 1e-10, 'kernel': C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0), 'n_restarts_optimizer': 20},
        1: {'alpha': 0.001, 'kernel': C(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0), 'n_restarts_optimizer': 5}
    }

    # Procesar cada movimiento ('Type'): 0 (going) y 1 (return)
    for movement_type in [0, 1]:
        # Filtrar por 'Type'
        type_data = data[data['Type'] == movement_type]

        # Extraer 'Name'
        names = type_data['Name']

        # Separar features
        X = type_data.drop(columns=['Name', 'Type', 'Kyu-Dan'])
        y = type_data['Kyu-Dan']

        # Separar los datos en entrenamiento y test
        random_state = 20 if dataset == 'bokken_data' else 42
        X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
            X, y, names, test_size=0.2, random_state=random_state, stratify=y
        )

        # Balancear los datos de entrenamiento con SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Escalado de características
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled  = scaler.transform(X_test)

        # Recuperar los mejores hiperparámetros para el movimiento actual
        params = optimal_params[movement_type]

        # Crear el modelo GPR con los mejores parámetros
        gpr = GaussianProcessRegressor(
            kernel=params['kernel'],
            alpha=params['alpha'],
            n_restarts_optimizer=params['n_restarts_optimizer']
        )

        # Entrenar el modelo GPR
        gpr.fit(X_train_scaled, y_train_balanced)

        # Realizar predicciones con los datos de test
        y_test_pred = gpr.predict(X_test_scaled)

        # Clip predictions to the valid range of class labels (0, 1, 2)
        y_test_pred_class = y_test_pred.round().astype(int)
        y_test_pred_class = y_test_pred_class.clip(min=0, max=2)

        # Evaluar el modelo
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2  = r2_score(y_test, y_test_pred)
        accuracy = accuracy_score(y_test, y_test_pred_class)
        print(f'Type {movement_type}: MSE = {test_mse:.4f}, R² = {test_r2:.4f}, Accuracy = {accuracy:.4f}')

        # Guardar las predicciones
        predictions = pd.DataFrame({
            'Name': names_test.values,
            'Actual': y_test.values,
            'Predicted': y_test_pred_class
        })
        # Añadir la columna type
        predictions['Type'] = movement_type
        all_predictions.append(predictions)

        # Mostrar el reporte
        print(f'Classification Report for Type {movement_type}:')
        print(classification_report(
            y_test,
            y_test_pred_class,
            labels=[0, 1, 2],  # Explicitly specify the labels (0: Bajo, 1: Medio, 2: Alto)
            target_names=['Bajo', 'Medio' ,'Alto']
        ))

        # Visualizar las predicciones vs. valores actuales
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.8, label='Predictions')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Fit')
        plt.title(f'Predictions vs Actuals for Type {movement_type}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid()
        plt.show()

        # Visualizar la distribución balanceada de las clases de formación
        plt.figure(figsize=(8, 6))
        pd.Series(y_train_balanced).value_counts().sort_index().plot(kind='bar', alpha=0.8)
        plt.title(f'Balanced Class Distribution After SMOTE (Type {movement_type})')
        plt.xlabel('Kyu-Dan')
        plt.ylabel('Count')
        plt.xticks(
            ticks=[0, 1, 2],
            labels=['Bajo', 'Medio' ,'Alto'],
            fontsize=12
        )
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    # Concatenar las predicciones
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)

    predicted_pivot = all_predictions_df.pivot(
        index='Name',      # Índice será 'Name'
        columns='Type',    # Crear columnas basadas en 'Type'
        values='Predicted' # Usar los valores de 'Predicted'
    )

    predicted_pivot.columns = [f'Predicted_{int(col)}' for col in predicted_pivot.columns]
    result = all_predictions_df[['Name', 'Actual']].drop_duplicates().merge(
        predicted_pivot, on='Name', how='left'
    )

    # Resetear el índice para convertir 'Name' en una columna normal
    result_predictions = result.reset_index().dropna()

    result_predictions.to_csv(f'{dataset}_predictions.csv', index=False)

# TCN
class TCN(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=3, kernel_size=3):
        super(TCN, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=input_size if i == 0 else hidden_size,
                    out_channels=hidden_size,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
        self.tcn = nn.Sequential(*layers)
        self.fc  = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.tcn(x)
        x = torch.mean(x, dim=-1)  # Global Average Pooling
        x = self.fc(x)
        return x
def apply_tcn(dataset):
    # Cargar el dataset
    data = pd.read_csv(f'{dataset}_selected_features.csv', sep='|')

    # Inicializar StandardScaler
    scaler = StandardScaler()

    # Tipos de movimiento (0: going, 1: return)
    for movement_type in [0, 1]:
        # Filtrar por tipo de movimiento
        type_data = data[data['Type'] == movement_type]

        # Separar características (X) y etiquetas (y)
        X = type_data.drop(columns=['Name', 'Type', 'Kyu-Dan'])
        y = type_data['Kyu-Dan']

        # Dividir el dataset en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Balancear los datos de entrenamiento con SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Escalar las características
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled = scaler.transform(X_test)

        # Convertir los datos a tensores
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)
        y_train_tensor = torch.tensor(y_train_balanced, dtype=torch.long)
        X_test_tensor  = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
        y_test_tensor  = torch.tensor(y_test.values, dtype=torch.long)

        # Crear DataLoader para entrenamiento y prueba
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
        test_loader  = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32)

        # Configurar el modelo TCN
        input_size  = 1 # Una sola dimensión de tiempo
        num_classes = len(y.unique())
        model       = TCN(input_size, num_classes)

        # Configurar el optimizador y la función de pérdida
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Entrenar el modelo
        epochs = 20
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss   = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}')

        # Evaluar el modelo
        model.eval()
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                all_preds.append(torch.argmax(y_pred, dim=1).cpu())
                all_labels.append(y_batch.cpu())

        all_preds  = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Calcular la precisión
        accuracy = MulticlassAccuracy(num_classes=num_classes)
        acc      = accuracy(all_preds, all_labels)
        print(f'Type {movement_type}: Accuracy = {acc:.4f}')

        # Matriz de predicciones frente a valores reales
        plt.figure(figsize=(8, 6))
        plt.scatter(all_labels, all_preds, alpha=0.8, label='Predictions')
        plt.plot([all_labels.min(), all_labels.max()], [all_labels.min(), all_labels.max()], 'r--', label='Perfect Fit')
        plt.title(f'Predictions vs Actuals for Type {movement_type}')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid()
        plt.show()

# GBDT
class HyperTree(nn.Module):
    def __init__(self, input_size, output_size, embedding_size=32):
        super(HyperTree, self).__init__()
        # Generación de representaciones mediante un árbol
        self.tree_representation = nn.Linear(input_size, embedding_size)
        # Expansión de la representación para mapear a parámetros del modelo objetivo
        self.random_projection = nn.Linear(embedding_size, 64)  # Proyección a espacio más alto
        # Generar los parámetros del modelo objetivo (por ejemplo, AR(3) o ETS)
        self.parameter_mapping = nn.Linear(64, output_size)

    def forward(self, x):
        tree_embed      = self.tree_representation(x)
        projected_embed = torch.relu(self.random_projection(tree_embed))
        parameters      = self.parameter_mapping(projected_embed)

        return parameters
def apply_gbdt(dataset):
    # Cargar el dataset
    data = pd.read_csv(f'{dataset}_selected_features.csv', sep='|')

    # Inicializar StandardScaler
    scaler = StandardScaler()

    # Ajustes de hiperparámetros
    tuned_lr             = 5e-4
    tuned_embedding_size = 64
    tuned_epochs         = 50

    # Procesar cada tipo de movimiento ('Type'): 0 (going) y 1 (return)
    for movement_type in [0, 1]:
        # Filtrar datos por tipo de movimiento
        type_data = data[data['Type'] == movement_type]

        # Separar características (X) y etiquetas (y)
        X = type_data.drop(columns=['Name', 'Type', 'Kyu-Dan'])
        y = type_data['Kyu-Dan']

        # Convertir etiquetas categóricas a numéricas
        y_numeric = y.astype('category').cat.codes

        # Dividir el dataset en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
        )

        # Balancear los datos de entrenamiento con SMOTE
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Escalar las características
        X_train_scaled = scaler.fit_transform(X_train_balanced)
        X_test_scaled  = scaler.transform(X_test)

        # Convertir los datos a tensores
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_balanced, dtype=torch.float32)
        X_test_tensor  = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor  = torch.tensor(y_test.values, dtype=torch.float32)

        # Configurar el modelo Hyper-Tree con hiperparámetros ajustados
        input_size = X_train_tensor.shape[1]
        output_size = 3  # Número de parámetros del modelo objetivo (ejemplo: AR(3))
        model = HyperTree(input_size, output_size, embedding_size=tuned_embedding_size)

        # Configurar el optimizador y la función de pérdida con hiperparámetros ajustados
        optimizer = torch.optim.Adam(model.parameters(), lr=tuned_lr)
        criterion = nn.MSELoss()

        # Entrenar el modelo Hyper-Tree
        for epoch in range(tuned_epochs):
            model.train()
            optimizer.zero_grad()
            parameters = model(X_train_tensor)  # Parámetros generados
            # Usar los parámetros para predecir etiquetas (por simplicidad, un modelo lineal)
            y_pred = parameters[:, 0] + parameters[:, 1] * X_train_tensor[:, 0]  # Modelo AR(1)
            loss = criterion(y_pred, y_train_tensor)
            loss.backward()
            optimizer.step()
            print(f'Epoch {epoch + 1}/{tuned_epochs}, Loss: {loss.item():.4f}')

        # Evaluar el modelo
        model.eval()
        with torch.no_grad():
            parameters_test   = model(X_test_tensor)
            y_test_pred       = parameters_test[:, 0] + parameters_test[:, 1] * X_test_tensor[:, 0]  # Modelo AR(1)
            y_test_pred_class = torch.round(y_test_pred).clip(min=0, max=2).long().numpy()

        # Calcular las métricas adicionales
        precision = precision_score(y_test_tensor.numpy(), y_test_pred_class, average='weighted')
        recall    = recall_score(y_test_tensor.numpy(), y_test_pred_class, average='weighted')
        f1        = f1_score(y_test_tensor.numpy(), y_test_pred_class, average='weighted')

        # Mostrar resultados
        print(f'Type {movement_type}: Test Loss = {mean_squared_error(y_test_tensor.numpy(), y_test_pred.numpy()):.4f}')
        print(f'Type {movement_type}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1-Score = {f1:.4f}')

        # Mostrar matriz de confusión
        cm   = confusion_matrix(y_test_tensor.numpy(), y_test_pred_class, labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bajo', 'Medio', 'Alto'])
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix for Type {movement_type}')
        plt.show()

        # Visualizar las predicciones frente a los valores reales
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test_tensor.numpy(), y_test_pred.numpy(), alpha=0.8, label='Predictions')
        plt.plot([y_test_tensor.min(), y_test_tensor.max()], [y_test_tensor.min(), y_test_tensor.max()], 'r--', label='Perfect Fit')
        plt.title(f'Predictions vs Actuals for Type {movement_type} (Hyper-Tree)')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.grid()
        plt.show()

# Run
def run(dataset):
    prepare_data(dataset)
    apply_gpr(dataset)
    apply_tcn(dataset)
    apply_gbdt(dataset)

run(bokken_data)
run(shikko_data)