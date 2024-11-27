#!/usr/bin/env python
# coding: utf-8


import obspy
from obspy import UTCDateTime
import glob
from eqcorrscan.core.match_filter import match_filter
import sys
import os

template_paths=glob.glob('/home/yaolinm/ondemand/German_DAS/summary/8000hz/*_8000hz.mseed')
template_paths.sort()
templates = []
template_names =[]
for i in range(66):
    template_names.append('template' +str(i))

for template_file in template_paths:
    st=obspy.read(template_file)
    for tr in st:
        tr.stats.channel=''
        tr.stats.station=''
        templates.append(obspy.Stream(tr))

stream_paths=glob.glob('/scratch/zspica_root/zspica1/yaolinm/German/German_DAS_8000/final_detection/mseed/*.mseed')
stream_paths.sort()

def scan(stream_idx):
    fn=str(os.path.basename(stream_paths[stream_idx]))+'.txt'
    st=obspy.read(stream_paths[stream_idx])
    st.detrend()
    st.filter('highpass',freq=50)
    st.detrend()
    
    for tr in st:
        tr.stats.station=''
        tr.stats.channel=''
    st.normalize()
    
    for i in range(len(st)):
        f=open(fn,'a')
        detections=match_filter(template_names=template_names, template_list=templates, st=obspy.Stream(st[i]), threshold=9, threshold_type='MAD',trig_int=0.05, plot=False, cores=1)
        for master in detections:
            print(str(i),master.detect_time, master.template_name, master.threshold, master.detect_val, file=f)
        f.close()

array_n=int(sys.argv[1])
scan(array_n)
      

