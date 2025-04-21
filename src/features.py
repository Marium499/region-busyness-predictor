import collections.abc
from math import radians #, cos, sin, asin, sqrt
import logging

import numpy as np
import pandas as pd
import h3
from sklearn.preprocessing import LabelEncoder #, OrdinalEncoder
# from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FunctionTransformer

# from utils import numpy_to_df

logger = logging.getLogger(__name__)


class FeaturePipeline(BaseEstimator, TransformerMixin):

  def __init__(self, k, resolution, restaurants_ids, random_state=1, R = 6372.8):
    '''
    Feature pipeline for the region busyness predictor.
    Args:
      k: number of clusters
      resolution: h3 resolution
      restaurants_ids: dictionary with restaurant ids and their coordinates
      random_state: random state for reproducibility
      R: radius of the earth in kilometers (default is 6372.8)
    '''
    self.k = k
    self.resolution = resolution
    self.restaurants_ids = restaurants_ids
    self.random_state = random_state
    self.centroids = None
    self.R = R

  def fit(self, df, y=None):

    df_restaurants = pd.DataFrame([{'lat': v['lat'], 'lon': v['lon']} for v in self.restaurants_ids.values()])
    self.centroids = initiate_centroids(self.k, df_restaurants)
    return self
  
  def add_restuarant_id(self, df):  

    df['restaurant_id'] = [self.restaurants_ids.get('{}_{}'.format(a,b)).get('id', -1) #fallback for unseen ones
                         for a,b in zip(df.restaurant_lat, df.restaurant_lon)]
    return df
    
  def add_distance_features(self, df):

    df['dist_to_restaurant'] = calc_dist(df.courier_lat, df.courier_lon, df.restaurant_lat, df.restaurant_lon)
    df['avg_dist_to_restaurants'] = [avg_dist_to_restaurants(lat,lon, self.restaurants_ids) for lat,lon in zip(df.courier_lat, df.courier_lon)]
    df['Hdist_to_restaurant'] = calc_haversine_dist(df.courier_lat.tolist(), df.courier_lon.tolist(), df.restaurant_lat.tolist(), df.restaurant_lon.tolist(), self.R)
    df['avg_Hdist_to_restaurants'] = [avg_Hdist_to_restaurants(lat,lon, self.restaurants_ids) for lat,lon in zip(df.courier_lat, df.courier_lon)]
    logger.info(f'Added distance features.')
    return df
  
  def add_time_features(self, df):

    df['courier_location_timestamp']=  pd.to_datetime(df['courier_location_timestamp'], format= 'ISO8601', errors='coerce')
    df['order_created_timestamp'] = pd.to_datetime(df['order_created_timestamp'], format= 'ISO8601', errors='coerce')
    df['date_day_number'] = [d for d in df.courier_location_timestamp.dt.day_of_year]
    df['date_hour_number'] = [d for d in df.courier_location_timestamp.dt.hour]
    logger.info(f'Updated time features.')
    return df
  
  def add_clustering_features(self, df):

    assignation, assign_errors = centroid_assignation(df, self.centroids)
    df['clusters_embedding'] = assignation
    df['clusters_embedding_error'] = assign_errors
    logger.info(f'Added clustering features.')
    return df
  
  def add_h3_features(self, df):

    df['h3_index'] = [h3.latlng_to_cell(lat, lon, self.resolution) for lat, lon in zip(df.courier_lat, df.courier_lon)]
    logger.info(f'Added h3 clustering features.')
    return df
  
  def add_order_busyness_features(self, df):

    df['orders_busyness_by_h3_hour'] = get_order_busyness(df)
    df['restaurants_per_index'] = get_restaurants_per_h3_index(df)
    logger.info(f'Added order busyness features.')
    return df
  
  def transform(self, df):
    
    df = self.add_restuarant_id(df)
    df = self.add_distance_features(df)
    df = self.add_time_features(df)
    df = self.add_clustering_features(df)
    df = self.add_h3_features(df)
    df = self.add_order_busyness_features(df)
    return df


def get_transformed_df(df, restaurants_ids, k, resolution, R=6372.8):
  '''
  Run the feature pipeline
  Args:
    df: pandas dataframe
    restaurants_ids: dictionary with restaurant ids and their coordinates
    k: number of clusters
    resolution: h3 resolution
    mode: train or predict
    target_column: target column name (optional)
  Returns:
    df_transformed: pandas dataframe with transformed features
    features_pipeline: FeaturePipeline object
    preprocess: ColumnTransformer object
  '''

  df = df.copy()
  features_pipeline = FeaturePipeline(k=k, resolution=resolution, restaurants_ids=restaurants_ids, R=R)
  
  # # Encode categorical features using OridinalEncoder
  # cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
  # preprocess = ColumnTransformer([
  #     ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).set_output(transform='pandas'), cat_cols)
  # ], remainder='passthrough', verbose_feature_names_out=True).set_output(transform='pandas')
  # # df_transformed = preprocess.fit_transform(df)

  # # sklearn pipeline
  # full_pipeline = Pipeline([
  #     ('features_pipeline', features_pipeline),
  #     ('preprocess', preprocess),
  # ])

  full_pipeline = Pipeline([
      ('features_pipeline', features_pipeline),
      ('encoder', FunctionTransformer(func=Encoder))
  ])
  try:
    df_transformed = full_pipeline.fit_transform(df)
    logger.info(f'Transformed data after full pipeline: {df_transformed.head()}')
  except Exception as e:
    logger.error(f'Error in full pipeline: {e}')
    raise e
  return df_transformed, full_pipeline


def Encoder(df):
  '''
  Encode categorical features using LabelEncoder

  Args:
    df: pandas dataframe

  Returns:
    df: pandas dataframe with encoded categorical features
  '''
  columnsToEncode = list(df.select_dtypes(include=['category','object']))
  le = LabelEncoder()
  for feature in columnsToEncode:
      try:
          df[feature] = le.fit_transform(df[feature])
      except:
          logger.error(f'Error encoding in {feature}')
  return df


  

### HELPER AND GEOMETRY FUNCTIONS ###

def calc_dist(p1x, p1y, p2x, p2y):
  '''
  Calculate the euclidean distances to restaurants arrays.

  Args:
    p1x: x coordinates of the courier
    p1y: y coordinates of the courier
    p2x: x coordinates of the restaurants
    p2y: y coordinates of the restaurants
    
  Returns:
    dist: euclidean distance
  '''
  p1 = (p2x - p1x)**2
  p2 = (p2y - p1y)**2
  dist = np.sqrt(p1 + p2)
  return dist.tolist() if isinstance(p1x, collections.abc.Sequence) else dist

def avg_dist_to_restaurants(courier_lat,courier_lon, restaurants_ids):
  '''
  Calculates the avgerage distance to restaurants.

  Args:
    courier_lat: latitude of the courier
    courier_lon: longitude of the courier
    restaurants_ids: dictionary with restaurant ids and their coordinates

  Returns:
    avg_dist: average distance to restaurants
  '''
  return np.mean([calc_dist(v['lat'], v['lon'], courier_lat, courier_lon) for v in restaurants_ids.values()])

def calc_haversine_dist(lat1, lon1, lat2, lon2, R = 6372.8):
  '''
  Calculate the great circle distance in kilometers between two points on the earth (specified in decimal degrees)
  
  Args:
    lat1: latitude of the first point 
    lon1: longitude of the first point
    lat2: latitude of the second point
    lon2: longitude of the second point

  Returns:
    dist: great circle distance in kilometers
  '''

  #    #3959.87433  this is in miles.  For Earth radius in kilometers use 6372.8 km
  if isinstance(lat1, collections.abc.Sequence):
    dLat = np.array([radians(l2 - l1) for l2,l1 in zip(lat2, lat1)])
    dLon = np.array([radians(l2 - l1) for l2,l1 in zip(lon2, lon1)])
    lat1 = np.array([radians(l) for l in lat1])
    lat2 = np.array([radians(l) for l in lat2])
  else:
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

  a = np.sin(dLat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dLon/2)**2
  c = 2*np.arcsin(np.sqrt(a))
  dist = R*c
  return dist.tolist() if isinstance(lon1, collections.abc.Sequence) else dist

def avg_Hdist_to_restaurants(courier_lat,courier_lon, restaurants_ids):
  '''
  Calculates the average haversine distance to restaurants.

  Args:
    courier_lat: latitude of the courier
    courier_lon: longitude of the courier
    restaurants_ids: dictionary with restaurant ids and their coordinates

  Returns:
    avg_dist: average distance to restaurants
  '''

  return np.mean([calc_haversine_dist(v['lat'], v['lon'], courier_lat, courier_lon) for v in restaurants_ids.values()])

def initiate_centroids(k, df_source):
    '''
    Initiate the centroids for the clustering algorithm.
    Args:
        k: number of clusters
        df_source: dataframe with the source data
    Returns:
        centroids: dataframe with the centroids
    '''
    # Randomly select k points from the source data as initial centroids
    centroids = df_source.sample(k, random_state=42)
    return centroids

def centroid_assignation(df, centroids):
    '''
    Assign each point to the nearest centroid.
    Args:
        df: dataframe with the points
        centroids: dataframe with the centroids
    Returns:
        assignation: list with the index of the nearest centroid for each point
        assign_errors: list with the distance to the nearest centroid for each point
    '''
    assignation = []
    assign_errors = []

    for _, obs in df.iterrows():
        errors = [eucl_dist(c['lat'], c['lon'], obs['courier_lat'], obs['courier_lon']) for _, c in centroids.iterrows()]
        nearest_idx = int(np.argmin(errors))
        assignation.append(nearest_idx)
        assign_errors.append(errors[nearest_idx])
    return assignation, assign_errors

def eucl_dist(p1x, p1y, p2x, p2y):
        return calc_dist(p1x, p1y, p2x, p2y)

  
def get_order_busyness(df): 
  '''
  Calculates the order busyness by h3 index and hour.

  Args:
    df: pandas dataframe

  Returns:
    order_busyness: list of order busyness by h3 index and hour
  '''
  index_list = [(i,d,hr) for (i,d,hr) in zip(df.h3_index, df.date_day_number, df.date_hour_number)]
  set_indexes = list(set(index_list))
  dict_indexes = {label: index_list.count(label) for label in set_indexes}
  order_busyness = [dict_indexes[i] for i in index_list]
  return order_busyness

def get_restaurants_per_h3_index(df):
  '''
  Calculates the number of restaurants per h3 index.

  Args:
    df: pandas dataframe

  Returns:
    restaurants_per_index: list of number of restaurants per h3 index
  '''
  restaurants_counts_per_h3_index = {a:len(b) for a,b in zip(df.groupby('h3_index')['restaurant_id'].unique().index, df.groupby('h3_index')['restaurant_id'].unique()) }
  restaurants_per_index = [restaurants_counts_per_h3_index[h] for h in df.h3_index]
  return restaurants_per_index

