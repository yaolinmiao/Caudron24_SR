#!/usr/bin/env python
# coding: utf-8

import numpy as np
import obspy
import glob
import os
from obspy import UTCDateTime as UTC
from scipy import signal
from obspy.signal.trigger import classic_sta_lta,trigger_onset,plot_trigger
import json
from nptdms import *
import csv
import matplotlib.pyplot as plt
import pickle
import sys


def locate_average(data,length,mode='maximum'):
    
    ave=[]
    for i in range(len(data)-length+1):
        ave.append(np.mean(data[i:length+i]))
        
    if mode =='maximum':
        return np.max(ave),np.argmax(ave)
    elif mode =='minimum':
        return np.min(ave),np.argmin(ave)
    else: 
        raise 'error'
        
def snr(data,st,lt,fs):
    signal=locate_average(np.abs(data),int(st*fs),'maximum')[0]
    noise=locate_average(np.abs(data),int(lt*fs),'minimum')[0]
    snr=signal/noise
    return snr

def check_existence(list1,list2,delta):
    for l2 in list2:
        c=0
        for l1 in list1:
            if np.abs(l1-l2)<=delta:
                break
            else:
                c+=1
        if c == len(list1):
            list1.append(l2)
    return list1

def sort_TM(file):
    
    f=open(file,'r')
    data=f.readlines()
    time=[]
    events=[]
    if len(data)>0:
        for i in range(len(data)):
            time.append(UTC(data[i].split(' ')[0]))
    events.append(time[0])
    
    events=check_existence(events,time,0.1)
    f.close()
    return events
    
master_dirs=np.sort(glob.glob('/scratch/zspica_root/zspica1/yaolinm/German/German_DAS_8000/final_detection/template_matching/*/'))
event_outdirs=np.sort(glob.glob('/scratch/zspica_root/zspica1/yaolinm/German/German_DAS_8000/final_detection/tm_results/events/*/'))
waveform_outdirs=np.sort(glob.glob('/scratch/zspica_root/zspica1/yaolinm/German/German_DAS_8000/final_detection/tm_results/waveforms/*/'))
stream_paths=glob.glob('/scratch/zspica_root/zspica1/yaolinm/German/German_DAS_8000/final_detection/mseed/*.mseed')
stream_paths.sort()

def extract_events(i,sec):
    
    ### i as stream index
    ### sec as section index
    
    master_dir=master_dirs[sec]
    name=os.path.basename(stream_paths[i])+'.txt'
    events=[]
    counts=np.zeros(len(glob.glob(master_dir+'*/')))
    
    for j in range(len(counts)):
        
        batchfiles=glob.glob(master_dir+'batch'+str(j+1).zfill(2)+'/*.txt')
        bname=master_dir+'batch'+str(j+1).zfill(2)+'/'+name

        if bname in batchfiles:

            file=bname

            if j==0 or np.sum(counts[:j])==0:
                events=sort_TM(file)
                counts[j]=len(events)

            else:               
                startlen=len(events)

                f=open(file,'r')
                data=f.readlines()
                time=[]

                for d in range(len(data)):
                    time.append(UTC(data[d].split(' ')[0]))

                events=check_existence(events,time,0.1)
                counts[j]=len(events)-startlen  

        else:
            pass
 
    outdir=event_outdirs[sec]
    np.save(outdir+name[:-10]+'.npy',counts)
    with open(outdir+name[:-10]+'.txt','wb') as fp:
        pickle.dump(events,fp)
           
    return events

def trim_mseed(events,i,sec):
    
    if sec==0:
        st=obspy.read(stream_paths[i])[130:170]
    elif sec==1:
        st=obspy.read(stream_paths[i])[220:260]
    elif sec==2:
        st=obspy.read(stream_paths[i])[260:300]
    else:
        st=obspy.read(stream_paths[i])[590:630]
    
    st.filter('highpass',freq=50)
    
    for a0 in events:
        
        a0=UTC(a0)       
    
        if st[0].stats.starttime<=a0-0.05 and a0+0.15<=st[0].stats.endtime:
            stime=a0-0.05
            newst=st.slice(stime,stime+0.2)
            for tr in newst:
                tr.data=tr.data.astype(np.int16)
            title=(str(newst[0].stats.starttime)[5:7]+'_'+str(newst[0].stats.starttime)[8:10]+'_'
                           +str(newst[0].stats.starttime)[11:13]+'_'+str(newst[0].stats.starttime)[14:16]+'_'
                           +str(newst[0].stats.starttime)[17:19]+'_'+str(newst[0].stats.starttime)[20:23])
            newst.write(waveform_outdirs[sec]+title+'.mseed')

            maxes=[]
            trids=[]
            for t in range(len(newst)):
                tr=newst[t]
                trsnr=snr(tr.data,0.005,0.05,tr.stats.sampling_rate)
                if trsnr >=5:
                    maxes.append(locate_average(np.abs(tr.data),int(0.005*tr.stats.sampling_rate),'maximum')[1])
                    trids.append(t)

            if len(maxes) >=15:
                max_whole=np.median(maxes)
                peaktime=stime+max_whole/newst[0].stats.sampling_rate

                if peaktime-0.03>= stime and peaktime+0.07 <=stime+0.2:
                    newst.trim(peaktime-0.03,peaktime+0.07)
                elif peaktime-0.03<stime:
                    newst.trim(stime,stime+0.1)
                else:
                    newst.trim(stime+0.1,stime+0.2)
                produce(newst)
   
    
def produce(waveform):
    
    waveform.normalize()
    
    ccc=int(waveform[0].stats.channel)
    
    mat=np.zeros((len(waveform),401))
    for ii in range(len(waveform)):
        f,pe=signal.periodogram(waveform[ii].data,fs=8000)
        mat[ii,:]=np.log10(pe[:401])
    title=(str(waveform[0].stats.starttime)[5:7]+'_'+str(waveform[0].stats.starttime)[8:10]+'_'
           +str(waveform[0].stats.starttime)[11:13]+'_'+str(waveform[0].stats.starttime)[14:16]+'_'
           +str(waveform[0].stats.starttime)[17:19]+'_'+str(waveform[0].stats.starttime)[20:23]+'_ch'+str(ccc))
     
    np.save('/scratch/zspica_root/zspica1/yaolinm/German/German_DAS_8000/final_detection/tm_results/npys/'+title+'.npy',mat)
    
    fig=plt.figure(figsize=(8,5),dpi=300)
    fig.subplots_adjust(wspace=0.4)

    ax1 = plt.subplot2grid(shape=(4,8), loc=(0,0), colspan=3,rowspan=4)
    ax2 = plt.subplot2grid(shape=(4,8), loc=(0,3), colspan=2,rowspan=4)
    ax3 = plt.subplot2grid(shape=(4,8), loc=(0,5), colspan=3,rowspan=4)


    for ii in range(len(waveform)):
        ax1.plot(waveform[ii].data+2*ii,linewidth=0.5,c='k')
    ax1.set_ylim(-1,79.5)
    ax1.set_yticks([0,20,40,60,79])
    ax1.set_yticklabels(['Ch.'+str(ccc),'Ch.'+str(ccc+10),'Ch.'+str(ccc+20),'Ch.'+str(ccc+30),'Ch.'+str(ccc+40)])
    ax1.set_xticks([0,400,800])
    ax1.set_xticklabels([0,0.05,0.1],fontsize=10)
    ax1.set_xlabel('Seconds after '+str(waveform[0].stats.starttime)[11:23])
    ax1.set_xlim(0,800)

    ax2.set_xticks([0,100,200,300,400])
    ax2.set_xticklabels([0,1000,2000,3000,4000],fontsize=8)
    ax2.set_yticks([])
    ax2.set_xlabel('Frequency (Hz)')

    sc=ax2.imshow(np.flip(mat,axis=0),aspect='auto',vmin=-7,vmax=-4.5)
    ax2.set_ylim(39,0)

    ax3.set_yticks([])
    ax3.set_xlabel('Frequency (Hz)')

    for i in range(len(waveform)):
        ax3.plot(f[:401],10**mat[i,:]/np.max(10**mat[i,:])+2*i,c='k',linewidth=0.8)
    ax3.set_ylim(-1,79.5)
    ax3.set_xticks([0,1000,2000,3000,4000])
    ax3.set_xticklabels([0,1000,2000,3000,4000],fontsize=8)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_xlim(0,4000)


    cax=fig.add_axes([0.52,0.85,0.07,0.02])
    cbar=fig.colorbar(sc, cax=cax,orientation='horizontal')
    cbar.set_label('log10(PSD)',fontsize=8,c='w')
    cbar.ax.xaxis.set_tick_params(color='w')
    cbar.outline.set_edgecolor('w')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')
    
    plt.savefig('/scratch/zspica_root/zspica1/yaolinm/German/German_DAS_8000/final_detection/tm_results/pngs/'+title+'.png',dpi=fig.dpi)
    
idx=int(sys.argv[1])
for s in range(4):
    events=extract_events(idx,s)
    trim_mseed(events,idx,s)

