#!/usr/bin/env python
# coding: utf-8


import obspy
from obspy import UTCDateTime
import glob
from eqcorrscan.core.match_filter import match_filter
import os
import random
import math
import sys


stream_paths=glob.glob('/scratch/zspica_root/zspica1/yaolinm/German/German_DAS_8000/final_detection/mseed/*.mseed')
stream_paths.sort()

template_paths=glob.glob('/scratch/zspica_root/zspica1/yaolinm/German/German_DAS_8000/final_detection/templates/ch130_170/*')
template_paths.sort()


templates = []
template_names =[]
for i in range(len(template_paths)):
    
    template_file=template_paths[i]
    st=obspy.read(template_file)
    st.normalize()
    st.detrend()
    lens=[]
    for tr in st:
        lens.append(tr.stats.npts)
        
    if len(list(set(lens)))==1 and list(set(lens))[0]==801:
        templates.append(obspy.Stream(st))
        template_names.append(os.path.basename(template_paths[i]))


stream_idx=int(sys.argv[1])
master_dir='/scratch/zspica_root/zspica1/yaolinm/German/German_DAS_8000/final_detection/template_matching/ch130_170/batch'


st=obspy.read(stream_paths[stream_idx])
st.filter('highpass',freq=50)
st.normalize()
st.detrend()

for i in range(12):

    detections=match_filter(template_names=template_names[200*i:min(len(templates),200*i+200)], template_list=templates[200*i:min(len(templates),200*i+200)], st=st, threshold=9, threshold_type='MAD',trig_int=0.2, plot=False, cores=1)
    
    if len(detections) >0:
        fn=master_dir+str(i+1).zfill(2)+'/'+str(os.path.basename(stream_paths[stream_idx]))+'.txt'
        f=open(fn,'a')
        for master in detections:
            print(master.detect_time, master.template_name, master.threshold, master.detect_val, file=f)
        f.close()
    

    