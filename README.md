# Caudron24_SR
Core code for bubble detection, frequency analysis, and signal clustering
The detection part relies on template matching with the EQcorrscan package; the clustering part relies on training CNN models + K-means clustering on compressed bottleneck layer with the Keras package

first_TM: cross-sensor waveform cross-correlation, it can be skipped if working with only a few sensors and having enough templates
master_TM: conventional template matching
tm_summary: estimate duration, the dominant frequency of individual detected events
clustering: structure of CNN model, and train models in search for the optimal-accuracy model
