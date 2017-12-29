from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import ekg_data

from sklearn.cluster import KMeans


def sampling_data(data, leg_size, window_size):
# This function is for time series data sampling
# leg_size : Length of splited data
# window_size : Sliding window size
  samples = []
  for pos in range(0,len(data),window_size):
    sample = np.copy(data[pos:pos+leg_size])
    if len(sample) != leg_size:
      continue
    samples.append(sample)
  return samples

def plot_waves(waves,step):

# waves : waves to plot
# step : sampling rate to plot

  plt.figure()
  n_graph_rows = 3
  n_graph_cols = 3
  graph_n = 1
  wave_n = 0

  for _ in range(n_graph_rows):
    for _ in range(n_graph_cols):
      axes = plt.subplot(n_graph_rows, n_graph_cols, graph_n)
      axes.set_ylim([-100,150])
      plt.plot(waves[wave_n])
      graph_n += 1
      wave_n += step

  plt.tight_layout()
  plt.show()

def reconstruct(data, window, clusterer):
# reconstruct data with centeroid of clusterer

# data is input data
# window is window function that make (input data's starting point and ending point) to be zero
# Clusterer : scikit-learn Cluster model

  window_len = len(window)
  slide_len = window_len/2
  
# data spliting with sliding lenght window_len/2 ( some datas are overlapping )
  segments = sampling_data(data, window_len, slide_len)
  

  reconstructed_data = np.zeros(len(data))


# find nearest centroid among clusters for reconstruction
  for segment_n, segment in enumerate(segments):
    segment *= window
   
    segment = np.reshape(segment,(1,window_len))
    nearest_match_idx = clusterer.predict(segment)[0]
    nearest_match = np.copy(clusterer.cluster_centers_[nearest_match_idx])
  
    pos = segment_n*slide_len
    reconstructed_data[pos:pos+window_len] += nearest_match
  
  return reconstructed_data


if __name__ == "__main__":
  n_samples = 8192
  window_len = 32
  # to avoid too many data

  data = ekg_data.read_ekg_data('a02.dat')[0:n_samples]

# create anomalous data
  data_anomalous = np.copy(data)
  data_anomalous[210:215] = 0

# window function is sin^2
  window_rads = np.linspace(0,np.pi,window_len)
  window = np.sin(window_rads)**2

  segments = sampling_data(data,window_len,2)
  segments_anomalous = sampling_data(data_anomalous,window_len,2)

  windowed_segments = []

  for segment in segments:
    segment *= window
    windowed_segments.append(segment)


  print("Clustering...")
  clusterer = KMeans(n_clusters=150)
  clusterer.fit(windowed_segments)

  print("Reconstructing...")

  reconstruction = reconstruct(data_anomalous,window,clusterer)
  error = reconstruction - data



  print("Maximum error is ",max(error))

  plt.figure()
  n_plot_samples = 300
  plt.plot(data[0:n_plot_samples],label="Anomalous data")
  plt.plot(reconstruction[0:n_plot_samples],label="reconstruction")
  plt.plot(error[0:n_plot_samples],label="error")
  plt.legend()
  plt.show()



