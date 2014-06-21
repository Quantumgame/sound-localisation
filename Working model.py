#import pickle
import cPickle as pickle
from brian.globalprefs import *
import pygame
from brian import *
from brian.hears import *
set_global_preferences(useweave=True, brianhears_usegpu = True, openmp = True, magic_useframes = False)
from sklearn import svm
from sklearn import cross_validation
import time
from StringIO import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

sampletime = 0.5
averaging = 1050   # needs to be a factor of 0.5*44100 = 22050
samples = int(sampletime * 44100)

n_filters = 40
hrtfdb = IRCAM_LISTEN('ICAM/')
subject = 1002
hrtfset = hrtfdb.load_subject(subject)

n = 24
num_indices = hrtfset.num_indices

def colave(inpt, ncol):
    shp = list(inpt.shape)
    return reshape(mean(reshape(inpt, [inpt.size/ncol,ncol]),axis =1), [shp[0],shp[1]/ncol])

def rowave(inpt, nrow):
    shp = list(inpt.shape)
    return reshape(mean(reshape(inpt.T, [inpt.size/nrow,nrow]),axis =1).T, [shp[1],shp[0]/nrow]).T

def preprocess(x, samples, num_indices, n_filters, averaging):
    a = reshape(x, [samples*num_indices,2*n_filters])
    return rowave(a,averaging)

def bpm_to_index(bpm, sample_rate, start, bars):
    seconds_per_bar = 60.0/bpm*4
    start_index = int(seconds_per_bar*sample_rate*start)
    end_index = start_index + int(sample_rate*seconds_per_bar*bars)
    return (start_index, end_index)

def preprocess(soundlocation, hrtfset, n=6, samplerate = 44100, verbose=False, n_filters = 40, averaging = 1260, sampletime = 0.4, debug=False):
    samples = int(sampletime * 44100)
    num_indices = hrtfset.num_indices
    cfmin, cfmax, cfN = 150*Hz, 5*kHz, n_filters
    cf = erbspace(cfmin, cfmax, cfN)
    X = None
    y = None
    for i in range(0, n):
        if (verbose):
            starttime = time.time()
        offset = i*samples
        sound = Sound(soundlocation).left[offset:(offset+samples)]
        if (debug):
            print sampletime*i
            print offset

        hrtf_fb = hrtfset.filterbank(sound) # We apply the chosen HRTF to the sound, the output has 2 channels
        # We swap these channels (equivalent to swapping the channels in the subsequent filters, but simpler to do it with the inputs)
        swapped_channels = RestructureFilterbank(hrtf_fb, indexmapping=[1, 0]) 
        #Now we apply all of the possible pairs of HRTFs in the set to these swapped channels, which means repeating them num_indices times first    
        hrtfset_fb = hrtfset.filterbank(Repeat(swapped_channels, num_indices))
        # Now we apply cochlear filtering (logically, this comes before the HRTF
        # filtering, but since convolution is commutative it is more efficient to
        # do the cochlear filtering afterwards
        gfb = Gammatone(Repeat(hrtfset_fb, cfN),
                        tile(cf, hrtfset_fb.nchannels))
        # Half wave rectification and compression
        cochlea = FunctionFilterbank(gfb, lambda x:15*clip(x, 0, Inf)**(1.0/3.0))
        cochleaout = cochlea.process()
        if debug:
            print 'Cochlea shape:'
            print cochleaout.shape
        x = reshape(cochleaout, [samples*num_indices,2*n_filters])
        if debug:
            print 'x shape:'
            print x.shape
        if X == None:
            X = rowave(x,averaging) 
            y = repeat(range(0,num_indices),samples/averaging)
        else:
            X = vstack((X, rowave(x,averaging)))
            y = hstack((y, repeat(range(0,num_indices),samples/averaging)))
            
        if (verbose):
            endtime = time.time()
            print("Completed: %0.2f percent" % ((i+1.0)/n*100))
            print("Elapsed: %0.2f seconds. Est remaining time: %0.2f minutes" % ((endtime - starttime),(endtime - starttime)*(n-i) / 60))
    return X,y

print "Prepping data"
xave = None
yave = None

if (False):
    for i in range(0, n):
        starttime = time.time()
        offset = i*44100
        # This gives the number of spatial locations in the set of HRTFs
        num_indices = hrtfset.num_indices
        # A sound to test the model with
        #sound = Sound.whitenoise(500*ms)
        sound = Sound('sounds/playback Playback.wav').left[offset:(offset+samples)]
        # We apply the chosen HRTF to the sound, the output has 2 channels
        hrtf_fb = hrtfset.filterbank(sound)
        # We swap these channels (equivalent to swapping the channels in the
        # subsequent filters, but simpler to do it with the inputs)
        swapped_channels = RestructureFilterbank(hrtf_fb, indexmapping=[1, 0])
        # Now we apply all of the possible pairs of HRTFs in the set to these
        # swapped channels, which means repeating them num_indices times first
        hrtfset_fb = hrtfset.filterbank(Repeat(swapped_channels, num_indices))
        # Now we apply cochlear filtering (logically, this comes before the HRTF
        # filtering, but since convolution is commutative it is more efficient to
        # do the cochlear filtering afterwards
        cfmin, cfmax, cfN = 150*Hz, 5*kHz, n_filters
        cf = erbspace(cfmin, cfmax, cfN)
        # We repeat each of the HRTFSet filterbank channels cfN times, so that
        # for each location we will apply each possible cochlear frequency
        gfb = Gammatone(Repeat(hrtfset_fb, cfN),
                        tile(cf, hrtfset_fb.nchannels))
        # Half wave rectification and compression
        cochlea = FunctionFilterbank(gfb, lambda x:15*clip(x, 0, Inf)**(1.0/3.0))
        cochleaout = cochlea.process()
        x = reshape(cochleaout, [samples*num_indices,2*n_filters])
        if xave == None:
            xave = rowave(x,averaging) 
            yave = repeat(range(0,num_indices),samples/averaging)
        else:
            xave = vstack((xave, rowave(x,averaging)))
            yave = hstack((yave, repeat(range(0,num_indices),samples/averaging)))
        endtime = time.time()
        print("Completed: %0.2f percent" % ((i+1.0)/n*100))
        print("Elapsed: %0.2f seconds. Est remaining time: %0.2f minutes" % ((endtime - starttime),(endtime - starttime)*(n-i) / 60))
        savetxt('xave1.csv',xave,delimiter=",")
        savetxt('yave.csv',yave,delimiter=",")
    xfile = open(r'x.pkl','wb')
    pickle.dump(xave,xfile)
    xfile.close()
    yfile = open(r'y.pkl','wb')
    pickle.dump(yave,yfile)
    print xave.shape
    print yave.shape

with open('x.pkl') as xfile:
    xave = pickle.load(xfile)

with open('y.pkl') as yfile:
    yave = pickle.load(yfile)

print "Imported from pickle"
print xave.shape, yave.shape
# ### Random Forest

print "Fitting a random forest"
from sklearn.cross_validation import cross_val_score

clf = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=1, random_state=0, n_jobs=8)
scores = cross_val_score(clf, xave, yave)
print scores.mean()

print "Fitting Extra Randomised Trees"
# ### Extra Randomised Trees

# <codecell>

from sklearn.ensemble import ExtraTreesClassifier

clf_extra = ExtraTreesClassifier(n_estimators=110, max_depth=None,min_samples_split=1, random_state=0, n_jobs=8)
scores_extra = cross_val_score(clf_extra, xave, yave)
print scores_extra.mean()

#Xtest, ytest = preprocess('sounds/netlix.wav',hrtfset, averaging=1050, verbose=True, sampletime = 0.5, n=4)
#print ytest.shape
#print Xtest.shape
#clf_extra.predict(Xtest[0:100])
