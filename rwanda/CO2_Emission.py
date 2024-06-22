import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

import geopandas as gpd
from shapely.geometry import Point
import folium

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans

# Load Data
df_train = pd.read_csv("train.csv")
print(f"Train set shape:\n{df_train.shape}\n")

df_test = pd.read_csv("test.csv")
print(f"Test set shape:\n{df_test.shape}\n")

"""# **EDA - Pre Data Processing**"""

df_train.head()

df_train.info()

df_train.describe().T

# df_train no ID
df_train_nID = df_train.drop(columns=['ID_LAT_LON_YEAR_WEEK'])
#df_train_nID.info()

df_test.head()

df_test.info()

df_test.describe().T

"""## Histogram Awal Data Train"""

fig_ = df_train.hist(
    figsize=(40, 20),
    bins=50,
    color="seagreen",
    edgecolor="black",
    xlabelsize=8,
    ylabelsize=8
)

plt.tight_layout()
plt.show()

"""## Geographical Mapping"""

# Combine train and test for easy visualisation
train_coords = df_train.drop_duplicates(subset = ['latitude', 'longitude'])
test_coords = df_test.drop_duplicates(subset = ['latitude', 'longitude'])
train_coords['set_type'], test_coords['set_type'] = 'train', 'test'

all_data = pd.concat([train_coords, test_coords], ignore_index = True)
# Create point geometries

geometry = gpd.points_from_xy(all_data.longitude, all_data.latitude)
geo_df = gpd.GeoDataFrame(
    all_data[["latitude", "longitude", "set_type", "emission"]], geometry=geometry
)

# Preview the geopandas df
geo_df.head()

geo_df.shape

# Jumlah titik lokasi
geo_df.geometry.nunique()

# Membuat kanvas untuk menaruh peta
all_data_map = folium.Map(prefer_canvas=True)

# Membuat list geometry dari GeoDataFrame
geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in geo_df.geometry]

# Iterasikan dengan list dan menambah tanda untuk tiap lokasi, dengan warna berdasarkan jenis data (train/test)
i = 0
for coordinates in geo_df_list:
    # Mengatur warna untuk data test dan train
    if geo_df.set_type[i] == "test":
        type_color = "orange"
    elif geo_df.set_type[i] == "train":
        type_color = "crimson"

    # Menempatkan penanda
    all_data_map.add_child(
        folium.CircleMarker(
            location=coordinates,
            radius = 1,
            weight = 4,
            zoom =10,
            popup=
            "Set: " + str(geo_df.set_type[i]) + "<br>"
            "Coordinates: " + str([round(x, 2) for x in geo_df_list[i]]),
            color =  type_color),
        )
    i = i + 1
all_data_map.fit_bounds(all_data_map.get_bounds())
all_data_map

"""Warna data train tidak terlihat karena lokasi pengambilan data train dan data test sama. Jadi, warnanya tumpang tindih.

## 10 Lokasi dengan emisi terbanyak (data train)
"""

# Menghitung jumlah emisi CO2 untuk setiap lokasi
total_emission_per_location = df_train.groupby(['latitude', 'longitude'])['emission'].sum().reset_index()

# Menampilkan hasil
df_total = total_emission_per_location.copy()
#df_total.head(10)

geometry = gpd.points_from_xy(df_total.longitude, df_total.latitude)
geo_df_train = gpd.GeoDataFrame(
    df_total[["latitude", "longitude", "emission"]], geometry=geometry
)

geo_df_train.head()

geo_df_train.describe().T

# Menghitung kuartil ke-3 dari kolom 'emission'
q3_emission = geo_df_train['emission'].quantile(0.75)
# Memfilter data yang nilai 'emission'-nya lebih besar dari Q3
max_emission = geo_df_train[geo_df_train['emission'] > q3_emission]

max_emission.describe().T

# Menghitung kuartil ke-1 dari kolom 'emission'
q1_emission = geo_df_train['emission'].quantile(0.25)
# Memfilter data yang nilai 'emission'-nya lebih kecil dari Q1
min_emission = geo_df_train[geo_df_train['emission'] < q1_emission]

min_emission.describe().T

# Membuat kanvas untuk menaruh peta
total_map = folium.Map(prefer_canvas=True)

# Membuat list geometry dari GeoDataFrame
geo_df_train_list = [[point.xy[1][0], point.xy[0][0]] for point in geo_df_train.geometry]

# Menentukan titik geometri lokasi dengan emisi terbanyak dan terendah
max_emission_location = [[point.xy[1][0], point.xy[0][0]] for point in max_emission.geometry]
min_emission_location = [[point.xy[1][0], point.xy[0][0]] for point in min_emission.geometry]

# Iterasikan dengan list dan menambah tanda untuk tiap lokasi, dengan warna berdasarkan jenis data (train/test)
i = 0
for coordinates in geo_df_train_list:
  type_color = 'orange'
  if coordinates in max_emission_location:
    type_color = 'red'
  elif coordinates in min_emission_location:
    type_color = 'purple'
  # Menempatkan penanda
  total_map.add_child(
      folium.CircleMarker(
      location=coordinates,
      radius = 1,
      weight = 4,
      zoom =10,
      popup=
      "Coordinates: " + str([round(x, 2) for x in geo_df_train_list[i]]),
      color =  type_color))
i = i + 1
total_map.fit_bounds(total_map.get_bounds())
total_map

max_emission

min_emission

"""Analisis:

Titik-titik lokasi emisi terbanyak dan tersedikit berdasarkan Q3 dan Q1 kolom 'emission' dapat dilihat pada peta. Dimana titik berwarna merah merupakan titik lokasi dengan emisi terbanyak, dan titik berwarna ungu merupakan titik-titik lokasi dengan emisi terendah.

Dapat dilihat bahwa titik lokasi di Rwanda dengan emisi yang sedikit berada di wilayah countryside / daerah pinggiran. Dan daerah dengan emisi terbanyak berada di pertengahan negara / daerah perkotaan, termasuk Capital City / Ibu Kota dari Rwanda,  yaitu Kigali.

## Tren Emisi
"""

# Tren Tahunan
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_train, x='year', y='emission', color='darkslateblue', errorbar=None)
plt.title('CO2 Emission Trends Over Years')
plt.xlabel('Year')
plt.ylabel('CO2 Emission')
plt.xticks([2019, 2020, 2021])
plt.show()

"""Emisi menurun drastis saat tahun 2020. Hal ini mungkin terjadi karena adanya lockdown akibat Covid-19."""

# Tren Mingguan tiap Tahun
plt.figure(figsize=(17, 8))

# Plotkan tren mingguan emisi CO2 untuk masing-masing tahun
sns.lineplot(data=df_train, x='week_no', y='emission', hue='year', marker='o')

plt.title('Weekly CO2 Emission Trends Over Years')
plt.xlabel('Week Number')
plt.ylabel('CO2 Emission')
plt.legend(title='Year')
plt.xticks(range(0, 52))
plt.show()

"""# **DATA PREPROCESSING**

## Handling Missing Values
"""

# Mencari Missing Value pada setiap kolom
missing_values = df_train.isnull().sum()

# Tampilkan kolom yang memiliki missing value
print("Kolom dengan Missing Value:")
print(missing_values[missing_values > 0])

# Menghitung persentase/perbandingan jumlah nilai null dan non-null

# Inisialisasi list untuk menyimpan hasil perhitungan
results = []

for col in df_train.columns:

  null_counts = df_train[col].isnull().sum()
  nonnull_counts = df_train[col].shape[0] - null_counts

  null_percentages = (null_counts / df_train[col].shape[0]) * 100
  nonnull_percentages = 100 - null_percentages

  # Tambahkan hasil perhitungan ke dalam list
  results.append({
      'Kolom': col,
      'Jumlah Null': null_counts,
      'Jumlah Non-Null': nonnull_counts,
      'Persentase Null': null_percentages,
      'Persentase Non-Null': nonnull_percentages
  })
# Buat DataFrame dari list hasil perhitungan
null_percentages_df = pd.DataFrame(results)
# Cetak DataFrame baru dalam bentuk tabel
print(null_percentages_df)

"""Parameter yang kolom-kolomnya memiliki missing value > 20%:
- NitrogenDioxide (NO2): 23.18%
- UvAerosolLayerHeight: 99.44%

Jika dilihat dari kebutuhan data dan banyaknya missing value, parameter NitrogenDioxide cukup penting dan jumlah missing valuenya masih bisa ditolerir. Untuk UvAerosolLayerHeight, jumlah missing value sangat besar dan dapat mempengaruhi model prediksi nantinya. Sehingga, kolom ini akan di drop dan tidak digunakan dalam pemodelan.

"""

# Drop columns containing 'UvAerosollLayerHeight'
df_train_new = df_train.drop(columns=df_train.columns[df_train.columns.str.contains('UvAerosolLayerHeight')])

df_train_new.info()

# Filling missing values with mean
numeric_columns = df_train_new.select_dtypes(include=['number']).columns
for col in numeric_columns:
    if col != 'ID_LAT_LON_YEAR_WEEK':
        mean = df_train_new[col].mean()
        df_train_new[col].fillna(mean, inplace=True)

"""## Feature Selection"""

# Data Train
main_columns_train = df_train_new[['SulphurDioxide_SO2_column_number_density',
                           'CarbonMonoxide_CO_column_number_density',
                           'CarbonMonoxide_H2O_column_number_density',
                           'NitrogenDioxide_NO2_column_number_density',
                           'NitrogenDioxide_tropospheric_NO2_column_number_density',
                           'NitrogenDioxide_stratospheric_NO2_column_number_density',
                           'Formaldehyde_tropospheric_HCHO_column_number_density',
                           'UvAerosolIndex_absorbing_aerosol_index',
                           'Ozone_O3_column_number_density',
                           'Cloud_cloud_fraction']]


df_train_new = pd.concat([df_train_new[['ID_LAT_LON_YEAR_WEEK', 'latitude','longitude','week_no']], main_columns_train, df_train_new['emission']], axis=1)
df_train_new.head()

df_train_new.info()

# Data Test
main_columns_test = df_test[['SulphurDioxide_SO2_column_number_density',
                           'CarbonMonoxide_CO_column_number_density',
                           'CarbonMonoxide_H2O_column_number_density',
                           'NitrogenDioxide_NO2_column_number_density',
                           'NitrogenDioxide_tropospheric_NO2_column_number_density',
                           'NitrogenDioxide_stratospheric_NO2_column_number_density',
                           'Formaldehyde_tropospheric_HCHO_column_number_density',
                           'UvAerosolIndex_absorbing_aerosol_index',
                           'Ozone_O3_column_number_density',
                           'Cloud_cloud_fraction']]


df_test_new = pd.concat([df_test[['ID_LAT_LON_YEAR_WEEK', 'latitude','longitude','week_no']], main_columns_test], axis=1)
df_test_new.head()

df_test_new.info()

"""## Outlier (run ulang sampai tidak ada lagi outlier)"""

# Mencari Outlier dengan Z-Score
z_scores = np.abs(stats.zscore(df_train_new.select_dtypes(include=np.number)))

threshold = 3
outliers = (z_scores > threshold).any(axis=0)
num_outlier_columns = sum(outliers)
print(f"Jumlah kolom yang memiliki outlier: {num_outlier_columns}")

# Menampilkan nama-nama kolom yang memiliki outlier
outlier_columns = df_train_new.select_dtypes(include=np.number).columns[outliers]
print("Kolom yang memiliki outlier:")
print(outlier_columns)

# Menghitung Z-score untuk setiap nilai dalam dataset
z_scores = np.abs(stats.zscore(df_train_new.select_dtypes(include=np.number)))
threshold = 3

# Mengecek jumlah outlier pada setiap kolom
outlier_count = (z_scores > threshold).sum(axis=0)
print("Jumlah outlier pada setiap kolom:")
print(outlier_count)

# Kolom SO2
kolom_SO2 = 'SulphurDioxide_SO2_column_number_density'

# Menghitung Z-Score untuk kolom yang dipilih
z_scores = np.abs(stats.zscore(df_train_new[kolom_SO2]))
threshold = 3
outliers = (z_scores > threshold)
num_outliers = sum(outliers)
print(f"Jumlah outlier pada kolom '{kolom_SO2}': {num_outliers}")

# Drop baris dengan outlier pada kolom secara iteratif
while num_outliers > 0:
    df_train_new = df_train_new[~outliers]
    z_scores = np.abs(stats.zscore(df_train_new[kolom_SO2]))
    outliers = (z_scores > threshold)
    num_outliers = sum(outliers)
    print(f"Jumlah baris outlier yang dihapus pada kolom '{kolom_SO2}': {num_outliers}")

# Cek Ulang SO2
z_scores_filtered = np.abs(stats.zscore(df_train_new[kolom_SO2]))
outliers_filtered = (z_scores_filtered > threshold)
num_outliers_filtered = sum(outliers_filtered)
if num_outliers_filtered > 0:
    print(f"Masih ada {num_outliers_filtered} baris yang merupakan outlier setelah drop.")
else:
    print(f"Outlier pada kolom '{kolom_SO2}' berhasil dihapus semua.")

# Kolom CO
kolom_CO = 'CarbonMonoxide_CO_column_number_density'

# Menghitung Z-Score untuk kolom yang dipilih
z_scores = np.abs(stats.zscore(df_train_new[kolom_CO]))
threshold = 3
outliers = (z_scores > threshold)
num_outliers = sum(outliers)
print(f"Jumlah outlier pada kolom '{kolom_CO}': {num_outliers}")

# Drop baris dengan outlier pada kolom secara iteratif
while num_outliers > 0:
    df_train_new = df_train_new[~outliers]
    z_scores = np.abs(stats.zscore(df_train_new[kolom_CO]))
    outliers = (z_scores > threshold)
    num_outliers = sum(outliers)
    print(f"Jumlah baris outlier yang dihapus pada kolom '{kolom_CO}': {num_outliers}")

## Cek Ulang CO
z_scores_filtered = np.abs(stats.zscore(df_train_new[kolom_CO]))
outliers_filtered = (z_scores_filtered > threshold)
num_outliers_filtered = sum(outliers_filtered)
if num_outliers_filtered > 0:
    print(f"Masih ada {num_outliers_filtered} baris yang merupakan outlier setelah drop.")
else:
    print(f"Outlier pada kolom '{kolom_CO}' berhasil dihapus semua.")

# Kolom H2O
kolom_H2O = 'CarbonMonoxide_H2O_column_number_density'

# Menghitung Z-Score untuk kolom yang dipilih
z_scores = np.abs(stats.zscore(df_train_new[kolom_H2O]))
threshold = 3
outliers = (z_scores > threshold)
num_outliers = sum(outliers)
print(f"Jumlah outlier pada kolom '{kolom_H2O}': {num_outliers}")

# Drop baris dengan outlier pada kolom secara iteratif
while num_outliers > 0:
    df_train_new = df_train_new[~outliers]
    z_scores = np.abs(stats.zscore(df_train_new[kolom_H2O]))
    outliers = (z_scores > threshold)
    num_outliers = sum(outliers)
    print(f"Jumlah baris outlier yang dihapus pada kolom '{kolom_H2O}': {num_outliers}")

## Cek Ulang H2O
z_scores_filtered = np.abs(stats.zscore(df_train_new[kolom_H2O]))
outliers_filtered = (z_scores_filtered > threshold)
num_outliers_filtered = sum(outliers_filtered)
if num_outliers_filtered > 0:
    print(f"Masih ada {num_outliers_filtered} baris yang merupakan outlier setelah drop.")
else:
    print(f"Outlier pada kolom '{kolom_H2O}' berhasil dihapus semua.")

# Kolom NO2
kolom_NO2 = 'NitrogenDioxide_NO2_column_number_density'

# Menghitung Z-Score untuk kolom yang dipilih
z_scores = np.abs(stats.zscore(df_train_new[kolom_NO2]))
threshold = 3
outliers = (z_scores > threshold)
num_outliers = sum(outliers)
print(f"Jumlah outlier pada kolom '{kolom_NO2}': {num_outliers}")

# Drop baris dengan outlier pada kolom secara iteratif
while num_outliers > 0:
    df_train_new = df_train_new[~outliers]
    z_scores = np.abs(stats.zscore(df_train_new[kolom_NO2]))
    outliers = (z_scores > threshold)
    num_outliers = sum(outliers)
    print(f"Jumlah baris outlier yang dihapus pada kolom '{kolom_NO2}': {num_outliers}")

## Cek Ulang NO2
z_scores_filtered = np.abs(stats.zscore(df_train_new[kolom_NO2]))
outliers_filtered = (z_scores_filtered > threshold)
num_outliers_filtered = sum(outliers_filtered)
if num_outliers_filtered > 0:
    print(f"Masih ada {num_outliers_filtered} baris yang merupakan outlier setelah drop.")
else:
    print(f"Outlier pada kolom '{kolom_NO2}' berhasil dihapus semua.")

# Kolom TNO2
kolom_TNO2 = 'NitrogenDioxide_tropospheric_NO2_column_number_density'

# Menghitung Z-Score untuk kolom yang dipilih
z_scores = np.abs(stats.zscore(df_train_new[kolom_TNO2]))
threshold = 3
outliers = (z_scores > threshold)
num_outliers = sum(outliers)
print(f"Jumlah outlier pada kolom '{kolom_TNO2}': {num_outliers}")

# Drop baris dengan outlier pada kolom secara iteratif
while num_outliers > 0:
    df_train_new = df_train_new[~outliers]
    z_scores = np.abs(stats.zscore(df_train_new[kolom_TNO2]))
    outliers = (z_scores > threshold)
    num_outliers = sum(outliers)
    print(f"Jumlah baris outlier yang dihapus pada kolom '{kolom_TNO2}': {num_outliers}")

## Cek Ulang TNO2
z_scores_filtered = np.abs(stats.zscore(df_train_new[kolom_TNO2]))
outliers_filtered = (z_scores_filtered > threshold)
num_outliers_filtered = sum(outliers_filtered)
if num_outliers_filtered > 0:
    print(f"Masih ada {num_outliers_filtered} baris yang merupakan outlier setelah drop.")
else:
    print(f"Outlier pada kolom '{kolom_TNO2}' berhasil dihapus semua.")

# Kolom SNO2
kolom_SNO2 = 'NitrogenDioxide_stratospheric_NO2_column_number_density'

# Menghitung Z-Score untuk kolom yang dipilih
z_scores = np.abs(stats.zscore(df_train_new[kolom_SNO2]))
threshold = 3
outliers = (z_scores > threshold)
num_outliers = sum(outliers)
print(f"Jumlah outlier pada kolom '{kolom_SNO2}': {num_outliers}")

# Drop baris dengan outlier pada kolom secara iteratif
while num_outliers > 0:
    df_train_new = df_train_new[~outliers]
    z_scores = np.abs(stats.zscore(df_train_new[kolom_SNO2]))
    outliers = (z_scores > threshold)
    num_outliers = sum(outliers)
    print(f"Jumlah baris outlier yang dihapus pada kolom '{kolom_SNO2}': {num_outliers}")

## Cek Ulang SNO2
z_scores_filtered = np.abs(stats.zscore(df_train_new[kolom_SNO2]))
outliers_filtered = (z_scores_filtered > threshold)
num_outliers_filtered = sum(outliers_filtered)
if num_outliers_filtered > 0:
    print(f"Masih ada {num_outliers_filtered} baris yang merupakan outlier setelah drop.")
else:
    print(f"Outlier pada kolom '{kolom_SNO2}' berhasil dihapus semua.")

# Kolom HCHO
kolom_HCHO = 'Formaldehyde_tropospheric_HCHO_column_number_density'

# Menghitung Z-Score untuk kolom yang dipilih
z_scores = np.abs(stats.zscore(df_train_new[kolom_HCHO]))
threshold = 3
outliers = (z_scores > threshold)
num_outliers = sum(outliers)
print(f"Jumlah outlier pada kolom '{kolom_HCHO}': {num_outliers}")

# Drop baris dengan outlier pada kolom secara iteratif
while num_outliers > 0:
    df_train_new = df_train_new[~outliers]
    z_scores = np.abs(stats.zscore(df_train_new[kolom_HCHO]))
    outliers = (z_scores > threshold)
    num_outliers = sum(outliers)
    print(f"Jumlah baris outlier yang dihapus pada kolom '{kolom_HCHO}': {num_outliers}")

## Cek Ulang HCHO
z_scores_filtered = np.abs(stats.zscore(df_train_new[kolom_HCHO]))
outliers_filtered = (z_scores_filtered > threshold)
num_outliers_filtered = sum(outliers_filtered)
if num_outliers_filtered > 0:
    print(f"Masih ada {num_outliers_filtered} baris yang merupakan outlier setelah drop.")
else:
    print(f"Outlier pada kolom '{kolom_HCHO}' berhasil dihapus semua.")

# Kolom UV
kolom_UV = 'UvAerosolIndex_absorbing_aerosol_index'

# Menghitung Z-Score untuk kolom yang dipilih
z_scores = np.abs(stats.zscore(df_train_new[kolom_UV]))
threshold = 3
outliers = (z_scores > threshold)
num_outliers = sum(outliers)
print(f"Jumlah outlier pada kolom '{kolom_UV}': {num_outliers}")

# Drop baris dengan outlier pada kolom secara iteratif
while num_outliers > 0:
    df_train_new = df_train_new[~outliers]
    z_scores = np.abs(stats.zscore(df_train_new[kolom_UV]))
    outliers = (z_scores > threshold)
    num_outliers = sum(outliers)
    print(f"Jumlah baris outlier yang dihapus pada kolom '{kolom_UV}': {num_outliers}")

## Cek Ulang UV
z_scores_filtered = np.abs(stats.zscore(df_train_new[kolom_UV]))
outliers_filtered = (z_scores_filtered > threshold)
num_outliers_filtered = sum(outliers_filtered)
if num_outliers_filtered > 0:
    print(f"Masih ada {num_outliers_filtered} baris yang merupakan outlier setelah drop.")
else:
    print(f"Outlier pada kolom '{kolom_UV}' berhasil dihapus semua.")

# Kolom O3
kolom_O3 = 'Ozone_O3_column_number_density'

# Menghitung Z-Score untuk kolom yang dipilih
z_scores = np.abs(stats.zscore(df_train_new[kolom_O3]))
threshold = 3
outliers = (z_scores > threshold)
num_outliers = sum(outliers)
print(f"Jumlah outlier pada kolom '{kolom_O3}': {num_outliers}")

# Drop baris dengan outlier pada kolom secara iteratif
while num_outliers > 0:
    df_train_new = df_train_new[~outliers]
    z_scores = np.abs(stats.zscore(df_train_new[kolom_O3]))
    outliers = (z_scores > threshold)
    num_outliers = sum(outliers)
    print(f"Jumlah baris outlier yang dihapus pada kolom '{kolom_O3}': {num_outliers}")

## Cek Ulang O3
z_scores_filtered = np.abs(stats.zscore(df_train_new[kolom_O3]))
outliers_filtered = (z_scores_filtered > threshold)
num_outliers_filtered = sum(outliers_filtered)
if num_outliers_filtered > 0:
    print(f"Masih ada {num_outliers_filtered} baris yang merupakan outlier setelah drop.")
else:
    print(f"Outlier pada kolom '{kolom_O3}' berhasil dihapus semua.")

# Kolom CCF
kolom_CCF = 'Cloud_cloud_fraction'

# Menghitung Z-Score untuk kolom yang dipilih
z_scores = np.abs(stats.zscore(df_train_new[kolom_CCF]))
threshold = 3
outliers = (z_scores > threshold)
num_outliers = sum(outliers)
print(f"Jumlah outlier pada kolom '{kolom_CCF}': {num_outliers}")

# Drop baris dengan outlier pada kolom secara iteratif
while num_outliers > 0:
    df_train_new = df_train_new[~outliers]
    z_scores = np.abs(stats.zscore(df_train_new[kolom_CCF]))
    outliers = (z_scores > threshold)
    num_outliers = sum(outliers)
    print(f"Jumlah baris outlier yang dihapus pada kolom '{kolom_CCF}': {num_outliers}")

## Cek Ulang CCF
z_scores_filtered = np.abs(stats.zscore(df_train_new[kolom_CCF]))
outliers_filtered = (z_scores_filtered > threshold)
num_outliers_filtered = sum(outliers_filtered)
if num_outliers_filtered > 0:
    print(f"Masih ada {num_outliers_filtered} baris yang merupakan outlier setelah drop.")
else:
    print(f"Outlier pada kolom '{kolom_CCF}' berhasil dihapus semua.")

# Kolom emisi
kolom_emisi = 'emission'

# Menghitung Z-Score untuk kolom yang dipilih
z_scores = np.abs(stats.zscore(df_train_new[kolom_emisi]))
threshold = 3
outliers = (z_scores > threshold)
num_outliers = sum(outliers)
print(f"Jumlah outlier pada kolom '{kolom_emisi}': {num_outliers}")

# Drop baris dengan outlier pada kolom secara iteratif
while num_outliers > 0:
    df_train_new = df_train_new[~outliers]
    z_scores = np.abs(stats.zscore(df_train_new[kolom_emisi]))
    outliers = (z_scores > threshold)
    num_outliers = sum(outliers)
    print(f"Jumlah baris outlier yang dihapus pada kolom '{kolom_emisi}': {num_outliers}")

## Cek Ulang emisi
z_scores_filtered = np.abs(stats.zscore(df_train_new[kolom_emisi]))
outliers_filtered = (z_scores_filtered > threshold)
num_outliers_filtered = sum(outliers_filtered)
if num_outliers_filtered > 0:
    print(f"Masih ada {num_outliers_filtered} baris yang merupakan outlier setelah drop.")
else:
    print(f"Outlier pada kolom '{kolom_emisi}' berhasil dihapus semua.")

"""Kalau di cek ulang masih ada outlier di run aja semua di bagian outlier sampe Jumlah Kolom yang memiliki outlier : 0"""

# Cek ulang
z_scores = np.abs(stats.zscore(df_train_new.select_dtypes(include=np.number)))

threshold = 3
outliers = (z_scores > threshold).any(axis=0)
num_outlier_columns = sum(outliers)
print(f"Jumlah kolom yang memiliki outlier: {num_outlier_columns}")

# Menampilkan nama-nama kolom yang memiliki outlier
outlier_columns = df_train_new.select_dtypes(include=np.number).columns[outliers]
print("Kolom yang memiliki outlier:")
print(outlier_columns)

df_train_new.shape

df_train_new.info()

"""## Handling Skewed Data"""

# Kemiringan awal
t = sns.displot(df_train_new['Ozone_O3_column_number_density'])
skewness="Skewness: %.2f"%(df_train_new['Ozone_O3_column_number_density'].skew())
plt.legend(title=f'{skewness}')

# Metode Log Transform
log_ozone = df_train_new['Ozone_O3_column_number_density'].map(lambda i: np.log(i) if i>0 else 0)

t = sns.displot(log_ozone)
skewness="Skewness: %.2f"%(log_ozone.skew())
plt.legend()
plt.legend(title=f'Log Transform\n{skewness}')

# Metode Box Cox
boxcox_ozone = df_train_new['Ozone_O3_column_number_density'].map(lambda i: np.abs(i) if i <0 else (i+1 if i==0 else i)) # ini untuk mapping negative dan zero values karena boxcox harus positive.
boxcox_ozone = stats.boxcox(boxcox_ozone)
boxcox_ozone = pd.Series(boxcox_ozone[0])

t = sns.displot(boxcox_ozone)
skewness="Skewness: %.2f"%(boxcox_ozone.skew())
plt.legend(title=f'Box Cox Transform\n{skewness}')

# Metode Square Root
sqrt_ozone = df_train_new['Ozone_O3_column_number_density'].map(lambda i: np.sqrt(i))

t = sns.displot(sqrt_ozone)
skewness="Skewness: %.2f"%(sqrt_ozone.skew())
plt.legend(title=f'Square Root Transform\n{skewness}')

"""Metode yang paling efektif adalah Box Cox Transform. Dengan metode ini, skor skewneww berkurang dari -0.56 menjadi 0.00."""

df_train_new['Ozone_O3_column_number_density'] = boxcox_ozone

"""Data Final:
- df_train_new
- df_test_new

# **EDA - Post Data Processing**
"""

df_train_new.head()

df_test_new.head()

df_test_new.week_no.unique()

"""## Korelasi Tiap Parameter dengan Target (Emisi)"""

df_train_new_nID = df_train_new.drop(columns=['ID_LAT_LON_YEAR_WEEK'])
df_train_corr = df_train_new_nID.corr()['emission']
df_corr_pos = df_train_corr[df_train_corr > 0]
df_corr_neg = df_train_corr[df_train_corr < 0]

df_train_corr

# Positive Correlation
df_corr_pos

# Negative Correlation
df_corr_neg

corr = df_train_new_nID[list(df_train_corr.index)].corr()

plt.figure(figsize = (13,8))
sns.heatmap(corr, cmap='RdYlGn', annot = True, center = 0)
plt.title('Correlation Table', fontsize = 15)
plt.show()

# Ekstrak kolom korelasi dengan "emission"
corr_emission = corr['emission'].drop('emission')

# Plot sebagai heatmap
plt.figure(figsize=(5, 5))
sns.heatmap(corr_emission.to_frame(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation with Emission')
plt.show()

"""## Histogram Distribusi Data Train"""

# Plot the distribution of all the numerical data
fig_ = df_train_new.hist(
    figsize=(25, 13),
    bins=50,
    color="seagreen",
    edgecolor="black",
    xlabelsize=8,
    ylabelsize=8
)

plt.tight_layout()
plt.show()

"""## Histogram Kolom-Kolom Parameter (Data Train)"""

# Plot the distribution of all main columns (before data cleaning)
main_columns_train = df_train[['SulphurDioxide_SO2_column_number_density',
                           'CarbonMonoxide_CO_column_number_density',
                           'CarbonMonoxide_H2O_column_number_density',
                           'NitrogenDioxide_NO2_column_number_density',
                           'NitrogenDioxide_tropospheric_NO2_column_number_density',
                           'NitrogenDioxide_stratospheric_NO2_column_number_density',
                           'Formaldehyde_tropospheric_HCHO_column_number_density',
                           'UvAerosolIndex_absorbing_aerosol_index',
                           'Ozone_O3_column_number_density',
                           'Cloud_cloud_fraction']]

fig_ = main_columns_train.hist(
    figsize=(20, 10),
    bins=50,
    color="forestgreen",
    edgecolor="black",
    xlabelsize=8,
    ylabelsize=8
)

plt.tight_layout()
plt.show()

# Plot the distribution of all main columns (after data cleaning)
main_columns_train_new = df_train_new[['SulphurDioxide_SO2_column_number_density',
                           'CarbonMonoxide_CO_column_number_density',
                           'CarbonMonoxide_H2O_column_number_density',
                           'NitrogenDioxide_NO2_column_number_density',
                           'NitrogenDioxide_tropospheric_NO2_column_number_density',
                           'NitrogenDioxide_stratospheric_NO2_column_number_density',
                           'Formaldehyde_tropospheric_HCHO_column_number_density',
                           'UvAerosolIndex_absorbing_aerosol_index',
                           'Ozone_O3_column_number_density',
                           'Cloud_cloud_fraction']]

fig_ = main_columns_train_new.hist(
    figsize=(20, 10),
    bins=50,
    color="forestgreen",
    edgecolor="black",
    xlabelsize=8,
    ylabelsize=8
)

plt.tight_layout()
plt.show()

"""## Histogram Distribusi Data Test"""

# Plot the distribution of all the numerical data
fig_ = df_test_new.hist(
    figsize=(30, 15),
    bins=50,
    color="seagreen",
    edgecolor="black",
    xlabelsize=15,
    ylabelsize=8
)

plt.tight_layout()
plt.show()

"""# **MODELLING**

## Pemilihan Model
"""

import numpy as np

from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
)

from sklearn.metrics import (
  mean_squared_error, r2_score, mean_absolute_error
)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import scipy.stats as stats
import matplotlib.pyplot as plt

# Bagi dataset menjadi set pelatihan dan pengujian
X = df_train_new.drop(["emission", "ID_LAT_LON_YEAR_WEEK"], axis=1)
y = df_train_new["emission"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing
    random_state=42   # Random seed for reproducibility
)

# Untuk menentukan model terbaik dan cek akurasi
# Model terbaik = R² tertinggi dan RMSE terendah
# R²
def rsqr_score(test, pred):
    """Calculate R squared score

    Args:
        test -- test data
        pred -- predicted data

    Returns:
        R squared score
    """
    r2_ = r2_score(test, pred)
    return r2_


# RMSE
def rmse_score(test, pred):
    """Calculate Root Mean Square Error score

    Args:
        test -- test data
        pred -- predicted data

    Returns:
        Root Mean Square Error score
    """
    rmse_ = np.sqrt(mean_squared_error(test, pred))
    return rmse_


# Print the scores
def print_score(test, pred):
    """Print calculated score

    Args:
        test -- test data
        pred -- predicted data

    Returns:
        print the regressor name
        print the R squared score
        print Root Mean Square Error score
    """

    print(f"- Regressor: {regr.__class__.__name__}")
    print(f"R²: {rsqr_score(test, pred)}")
    print(f"RMSE: {rmse_score(test, pred)}\n")

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Ridge': Ridge(),
    'Lasso': Lasso(alpha=0.001),
    'ElasticNet': ElasticNet(alpha=0.001),
    'Random Forest Regressor': RandomForestRegressor(),
    'XGBoost Regressor': XGBRegressor()
}

# Functions to calculate metrics
def rsqr_score(test, pred):
    return r2_score(test, pred)

def rmse_score(test, pred):
    return np.sqrt(mean_squared_error(test, pred))

def print_score(model_name, test, pred):
    print(f"- Regressor: {model_name}")
    print(f"R²: {rsqr_score(test, pred):.4f}")
    print(f"RMSE: {rmse_score(test, pred):.4f}\n")

# Iterate over models, train each one, and print the scores
best_model = None
best_r2 = -np.inf
best_rmse = np.inf

for name, model in models.items():
    # Create a pipeline with imputer and regression model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Imputer to fill NaN with mean values
        ('regressor', model)  # Current regression model
    ])

    # Train model on X_train and y_train
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate scores
    r2 = rsqr_score(y_test, y_pred)
    rmse = rmse_score(y_test, y_pred)

    # Print scores
    print_score(name, y_test, y_pred)

    # Determine the best model based on R² and RMSE
    if r2 > best_r2 and rmse < best_rmse:
        best_model = name
        best_r2 = r2
        best_rmse = rmse

print(f"Best Model: {best_model}")
print(f"Best R²: {best_r2:.4f}")
print(f"Best RMSE: {best_rmse:.4f}")

"""Model yang cocok dalam memprediksi emisi rwanda adalah model **Random Forest Regression**

## Evaluasi Model
"""

# Define the model
random_forest = RandomForestRegressor(random_state=42)

# Create a pipeline with imputer and random forest regressor
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('regressor', random_forest)
])

# Train the model on X_train and y_train
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print the evaluation metrics
print(f"R²: {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

"""## Grafik Model"""

import matplotlib.pyplot as plt

# Asumsi y_pred_rf berisi prediksi dari model random forest regressor
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color='red', label='Ideal Fit')
plt.xlabel('Actual Emission')
plt.ylabel('Predicted Emission')
plt.title('Random Forest Regressor: Actual vs Predicted Emission')
plt.legend()
plt.show()

"""Jika garis ideal pada plot ini miring ke kanan menandakan bahwa nilai aktual dan nilai prediksi naik yang artinya bahwa model memiliki akurasi yang baik"""

import scipy.stats as stats
import matplotlib.pyplot as plt

residuals_rf = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals_rf, alpha=0.3)
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='red', linestyles='dashed')
plt.xlabel('Predicted Emission')
plt.ylabel('Residuals')
plt.title('Random Forest Regressor: Residuals vs Predicted Emission')
plt.show()

# Interpretasi fitur penting
importance = random_forest.feature_importances_
features = X.columns

# Membuat DataFrame untuk interpretasi fitur penting
df_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
df_importance = df_importance.sort_values(by='Importance', ascending=False)

# Plot fitur penting untuk visualisasi pengaruh
plt.figure(figsize=(10, 6))
plt.barh(df_importance['Feature'], df_importance['Importance'])
plt.title("Fitur Penting pada Prediksi Penyakit Jantung (Random Forest Regression)")
plt.xlabel("Importance Score")
plt.ylabel("Fitur")
plt.show()

# Tampilkan interpretasi fitur penting
print("Interpretasi Fitur Penting:")
print(df_importance)

# Simpan DataFrame ke dalam file CSV
df_train_new.to_csv('df_train_new.csv', index=False)
