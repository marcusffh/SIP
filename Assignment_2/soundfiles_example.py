# Introduction to working with sound in python

# we will introduce using ipython audio and soundfile,
# but there are many 


# we will be using sounds provided by the royalty free sound effects archive from Sonniss at Game Audio GDC
# https://sonniss.com/gameaudiogdc 

# and impulses provided by Nikolay Georgiev https://georgievsound.com/free-downloads/ 


# In[103]:


# make sure you have soundfile installed 
# python -m pip install soundfile

# Note that soundfile normalizes the wave to be between -1 and 1,
# other python modules might treat this differently
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve

# Let's load the first file we will work with
soundbyte, soundbyte_samplerate = sf.read('samples/laugh2.wav')

# soundbyte is now a 2D numpy array
print('The shape of our soundbyte is',soundbyte.shape)

# It has two channels in this case, one for each 'ear', left and right
# we can plot these waves 
Left = soundbyte[:,0]
Right = soundbyte[:,1]

plt.figure(figsize=(20,5))
plt.plot(Left, label='Left')
plt.plot(Right, label='Right')
plt.legend()


# In[104]:


# samplerate tells us how many of these samples should be played per second
print('the samplerate of our soundbyte is',soundbyte_samplerate)

print('This means that the soundbyte is ', soundbyte.shape[0]/soundbyte_samplerate, 'seconds long')


# In[105]:


# If you are using Ipython you can play the audio directly with the following:

from IPython.display import Audio, display

# notice that we transpose the soundbyte using .T
# to use it with Ipython Audio
sound = Audio(soundbyte.T, rate=soundbyte_samplerate, autoplay=True) 
sound


# In[75]:


# Otherwise you can save the file and play it using your favourite player
sf.write('example_write.wav', soundbyte, soundbyte_samplerate)


# In[114]:


# Now we can manipulate the sound in various ways
# the simplest is just to speed it up or slow it down
# by setting the sample rate accordingly
slow = Audio(soundbyte.T, rate=soundbyte_samplerate // 2, autoplay=False) 
fast = Audio(soundbyte.T, rate=soundbyte_samplerate * 2, autoplay=False)
display(slow), display(fast)
sf.write('slow.wav', soundbyte, soundbyte_samplerate // 2)
sf.write('fast.wav', soundbyte, soundbyte_samplerate * 2)


# In[115]:


# we can also add noise to the sound
noise = np.random.normal(loc=0,scale=0.1,size=len(soundbyte))
noisy = soundbyte + np.expand_dims(noise, 1)
display(Audio(noisy.T, rate=soundbyte_samplerate, autoplay=False))

sf.write('noisy.wav', noisy, soundbyte_samplerate)


# In[116]:


# or we could add a fade into it
fade = np.linspace(0,1,len(soundbyte))

faded = soundbyte * np.expand_dims(fade, 1)
display(Audio(0.5*faded.T, rate=soundbyte_samplerate, autoplay=False))
display(Audio(soundbyte.T, rate=soundbyte_samplerate, autoplay=False))

sf.write('faded.wav', faded, soundbyte_samplerate)


# In[117]:



plt.figure(figsize=(20,5))
plt.plot(fade)
plt.title('Linear Fade In')
plt.figure(figsize=(20,5))
plt.plot(faded)
plt.title('Faded')
plt.figure(figsize=(20,5))
plt.title('Original')
plt.plot(soundbyte)


# In[119]:


# or we could play it backwards
backwards = soundbyte[::-1]
plt.figure(figsize=(20,5))
plt.plot(backwards)
display(Audio(backwards.T, rate=soundbyte_samplerate, autoplay=False))
sf.write('backwards.wav', backwards, soundbyte_samplerate)