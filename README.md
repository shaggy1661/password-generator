# voice-recognition
# Install python_speech_features module
!pip install python_speech_features

# Import all modules
import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from pydub import AudioSegment
from python_speech_features import mfcc
from time import time

# Load the csv file into data frame
df = pd.read_csv('../input/common-voice/cv-valid-train.csv')


# Create two new data frames
df_male = df[df['gender']=='male']
df_female = df[df['gender']=='female']

# Find out the number of rows
print(df_male.shape)		
# output: (55029, 8) 

print(df_female.shape)		
# output: (18249, 8)

# Take only 300 male and 300 female data
df_male = df_male[:300]
df_female = df_female[:300]

# Define the audio path
TRAIN_PATH = '../input/common-voice/cv-valid-train/'

# The function to convert mp3 to wav
def convert_to_wav(df, m_f, path=TRAIN_PATH):
    srcs = []

    for file in tqdm(df['filename']):
        sound = AudioSegment.from_mp3(path+file)
        
		# Create new wav files based on existing mp3 files
        if m_f == 'male':
            sound.export('male-'+file.split('/')[-1].split('.')[0]+'.wav', format='wav')
        elif m_f == 'female':
            sound.export('female-'+file.split('/')[-1].split('.')[0]+'.wav', format='wav')
        
    return

# How to use the convert_to_wav() function
convert_to_wav(df_male, m_f='male')
convert_to_wav(df_female, m_f='female')


# Define a function to load the raw audio files
def load_audio(audio_files):
	# Allocate empty list for male and female voices
    male_voices = []
    female_voices = []

    for file in tqdm(audio_files):
        if file.split('-')[0] == 'male':
            male_voices.append(librosa.load(file))
        elif file.split('-')[0] == 'female':
            female_voices.append(librosa.load(file))
    
	# Convert the list into Numpy array
    male_voices = np.array(male_voices)
    female_voices = np.array(female_voices)
    
    return male_voices, female_voices

# How to use load_audio() function
male_voices, female_voices = load_audio(os.listdir())


# The function to extract audio features
def extract_features(audio_data):

	audio_waves = audio_data[:,0]
	samplerate = audio_data[:,1][1]
	
	features = []
	for audio_wave in tqdm(audio_waves):
		features.append(mfcc(audio_wave, samplerate=samplerate, numcep=26))
    
	features = np.array(features)
	return features

# Use the extract_features() function
male_features = extract_features(male_voices)
female_features = extract_features(female_voices)


# The function used to concatenate all audio features forming a long 2-dimensional array
def concatenate_features(audio_features):
    concatenated = audio_features[0]
    for audio_feature in tqdm(audio_features):
        concatenated = np.vstack((concatenated, audio_feature))
        
    return concatenated

# How the function is used
male_concatenated = concatenate_features(male_features)
female_concatenated = concatenate_features(female_features)

print(male_concatenated.shape) 		
# Output: (117576, 26)

print(female_concatenated.shape)	
# Output: (124755, 26)


# Concatenate male voices and female voices
X = np.vstack((male_concatenated, female_concatenated))

# Create labels
y = np.append([0] * len(male_concatenated), [1] * len(female_concatenated))

# Check whether X and y are already having the exact same length
print(X.shape)		
# Output: (242268, 26)

print(y.shape)		
# Output: (242268,)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)


# Initialize SVM model
clf = SVC(kernel='rbf')      

# Train the model
start = time()
clf.fit(X_train[:50000], y_train[:50000])
print(time()-start)						
# Output: 184.8018662929535 (seconds)

# Compute the accuracy score towards train data
start = time()
print(clf.score(X_train[:50000], y_train[:50000]))		
# Output: 0.78204

print(time()-start)						
# Output: 90.8693311214447 (seconds)

# Compute the accuracy score towards test data
start = time()
print(clf.score(X_test[:10000], y_test[:10000]))		
# Output: 0.7679

print(time()-start)						
# Output: 18.082067728042603 (seconds)


# Predict the first 10000 test data
svm_predictions = clf.predict(X_test[:10000])

# Create the confusion matrix values
cm = confusion_matrix(y_test[:10000], svm_predictions)

# Create the confusion matrix display
plt.figure(figsize=(8,8))
plt.title('Confusion matrix on test data')
sns.heatmap(cm, annot=True, fmt='d', 
            cmap=plt.cm.Blues, cbar=False, annot_kws={'size':14})
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# Performance comparison between different algorithms
index = ['SVM-RBF', 'SVM-Poly', 'SVM-Sigmoid', 'Logistic Regression']

# I record all the results below manually
values = [184.8, 137.0, 283.6, 0.7]

plt.figure(figsize=(12,3))
plt.title('Training duration (lower is better)')
plt.xlabel('Seconds')
plt.ylabel('Model')
plt.barh(index, values, zorder=2)
plt.grid(zorder=0)

for i, value in enumerate(values):
    plt.text(value+20, i, str(value)+' secs', fontsize=12, color='black',
             horizontalalignment='center', verticalalignment='center')

plt.show()


# set width of bar
barWidth = 0.25
    
index = ['SVM-RBF', 'SVM-Poly', 'SVM-Sigmoid', 'Logistic Regression']

# set height of bar
# I record all the results below manually
train_acc = [78.2, 74.8, 74.8, 65.8]
test_acc = [76.8, 74.3, 74.3, 65.8]
 
# Set position of bar on X axis
baseline = np.arange(len(train_acc))
r1 = [x + 0.125 for x in baseline]
r2 = [x + 0.25 for x in r1]
 
# Make the plot
plt.figure(figsize=(16,9))
plt.title('Model performance (higher is better)')
plt.bar(r1, train_acc, width=barWidth, label='Train', zorder=2)
plt.bar(r2, test_acc, width=barWidth, label='Test', zorder=2)
plt.grid(zorder=0)
 
# Add xticks on the middle of the group bars
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks([r + barWidth for r in range(len(train_acc))], index)

# Create text
for i, value in enumerate(train_acc):
    plt.text(i+0.125, value-5, str(value), fontsize=12, color='white',
             horizontalalignment='center', verticalalignment='center')
    
for i, value in enumerate(test_acc):
    plt.text(i+0.375, value-5, str(value), fontsize=12, color='white',
             horizontalalignment='center', verticalalignment='center')
    
plt.legend()
plt.show()
