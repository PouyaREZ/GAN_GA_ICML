from __future__ import print_function, division

# -*- coding: utf-8 -*-
"""
Created on Fri May 14 16:07:53 2021

Code for running experiments on applying C-GAN to the case study

sys.argv[1] = 0: percentage of EV integration with energy system
Look at lines 298 to 309 to understand the functionality of the other system arguments.


@author: Author
"""


# 1 Train the C-GAN and output the initial solutions

'''
Taken from https://github.com/eriklindernoren/Keras-GAN/tree/master/cgan

Modified by Author
Last modified on 5/27/2021
'''
import sys

# from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# from keras.datasets import mnist
# from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
# =============================================================================
# from tensorflow.config import list_physical_devices
# print("Num GPUs Available: ", len(list_physical_devices('GPU')))
# =============================================================================


import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import os

# from collections import deque




from datetime import datetime

# print("Current Time =", current_time)


'''

##############################################################################
##############################################################################
##############################################################################

'''


## 1.2 Main (Input Data and Train CGAN)

#  Set a name to distinguish the generated logs, visualizations, and models
run_name = 'EnergySystem_wo_EV' if sys.argv[1] == '0' else 'EnergySystem_w_EV'
# Determine the name of the experiment you want to run
experiment = '3'
result_file_name = 'Results_EV_'+sys.argv[1]+'_Percent.csv' # argv[1] should be 0 or 100
now = datetime.now()
current_time_string = now.strftime("%Y_%m_%d_%H_%M_%S_%f")

log_filename = 'output_'+sys.argv[1]+'_Percent_'+current_time_string+'.txt'
print('Name of the output file for this run:%s'%log_filename)



def append_list_to_log(array):
    with open(log_filename, "a") as f:
        for i, element in enumerate(array):
            f.write(str(element))
            f.write(',')




### 1.2.1 Input Data

# Constants for loading the dataset
max_site_area = 10**6
max_FAR = 5
max_gfa = max_site_area*max_FAR
error_tol = 1.15




# Constants for plotting the filtered plots
# LCC_Cutoff = 10 # float('inf') # k$/m2
CO2_Cutoff = float('inf') # T-CO2/m2




# Constants for evaluating the solutions
# Num_Sites = 4
Num_Buildings = 21
Num_Decision_Vars = Num_Buildings + 2
Num_Obj_Fxns = 2
Type_Max = 7
# len_of_indiv = 11 # After adding the plant location


LCC_Var = Num_Buildings+5 # Can be overwritten based on the input_data_reused signal from load_data
CO2_Var = Num_Buildings+6 # Can be overwritten based on the input_data_reused signal from load_data
LCC_w_o_EV_from_Grid_Var = Num_Buildings+7
# WalkScore_Var = Num_Sites+15
GFA_Var = Num_Buildings+7+Type_Max+3
FAR_Var = Num_Buildings+7+Type_Max+1
EV_Ratio_Var = Num_Buildings+3


CHPType_Var = Num_Buildings
ChillerType_Var = Num_Buildings+1


input_data_reused = False



# Helper functions

def parse_list_argv(argv, outputtype):
    n = len(argv)
    a = argv[1:n-1]
    a = a.split(',')
    if outputtype == 'int':
        return [int(item) for item in a]
    else:
        return [float(item) for item in a]



# Import and filter the file `filename` based on several conditions
def DF_Filter(filename): # Similar to the one in Plots_Paper_One.py               
    file = np.loadtxt(filename, dtype='float', delimiter=',')
    inputDF = pd.DataFrame(file)

    
    
    print('+++++ processing %s +++++\n'%(filename))
    
    print('Count duplicates:')
    condition1 = inputDF.duplicated() == True
    print(inputDF[condition1][GFA_Var].count())
    
    
    print('Count under the min GFA:') # Count non-trivial neighborhoods
    condition2 = inputDF[GFA_Var] <= 1/error_tol#<=647497/10
    print(inputDF[condition2][GFA_Var].count())
    
    
    print('Count over the max GFA:')
    condition3 = inputDF[GFA_Var] >= max_gfa*error_tol
    print(inputDF[condition3][GFA_Var].count())
    
    
    print('Count over the max Site GFA:')
    condition4 = inputDF[GFA_Var]/inputDF[FAR_Var] >= max_site_area*error_tol
    print(inputDF[condition4][GFA_Var].count())


    # Normalizing the LCC and CO2 objectives
    print('Normalizing the LCC and CO2 Obj Fxns')
    inputDF[LCC_Var] /= inputDF[GFA_Var] # Normalizing LCC ($/m2)
    inputDF[CO2_Var] /= inputDF[GFA_Var] # Normalizing CO2 (Tonnes/m2)


    # Filtering the inadmissible results
    Filtered = ~(condition1 | condition2 | condition3 | condition4)
    inputDF = inputDF[Filtered]
    print('Count of valid answers: %d out of %d'%(len(inputDF), len(file)))


    print("Lowest LCC considered in training data:%.2f"%np.min(inputDF[LCC_Var]))
    print("Highest LCC considered in training data:%.2f"%np.max(inputDF[LCC_Var]))
    print("Lowest CO2 considered in training data: %.2f"%np.min(inputDF[CO2_Var]))
    print("Highest CO2 considered in training data: %.2f"%np.max(inputDF[CO2_Var]))

    inputDF.reset_index(inplace=True, drop=True)
    
    return inputDF


# Convert the raw input data into training features and labels vectors
def load_data(input_data_reused):
    ## IMPORT DATA       
    print('loading data')
    input_data_address = 'input_data/'+run_name+'.txt'
    
    if not os.path.exists(input_data_address):
        filenames = [result_file_name]
        # DFNames = ['CCHP+Network']
        DF = DF_Filter(filenames[0])
    
        ## Set the range of input variables for the NN, then import the train variables
        inputRange = list(range(Num_Decision_Vars))
        # Define X
        X = DF.iloc[:,inputRange]
        # Define y
        outputRange = [LCC_Var, CO2_Var] # THESE ARE NORMALIZED IN DF_Filter
        y = DF.iloc[:, outputRange]
    else: # Data already filtered and saved, just reload it
        file = np.loadtxt(input_data_address)
        DF = pd.DataFrame(file)
        print('Loaded training data from %s'%input_data_address)
        inputRange = list(range(Num_Decision_Vars))
        X = DF.iloc[:,inputRange]
        outputRange = list(range(-Num_Obj_Fxns,0,1)) # THESE ARE NORMALIZED IN DF_Filter
        y = DF.iloc[:, outputRange]
        input_data_reused = True

    
    print('Scaling data')
    # Scale X
    x = X.values #returns a numpy array
    std_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_Scaler = std_scaler.fit(x)
    X = X_Scaler.transform(x)
    # X = std_scaler.fit_transform(x)    
    
    # Scale y
    y_values = y.values #returns a numpy array
    std_scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    y_Scaler = std_scaler2.fit(y_values)
    y = y_Scaler.transform(y_values)

    # Experiment: not scaling y: FAILED

    X_test = None
    y_test = None
    X_train = X
    y_train = y
    
    ## Reshape the data to fit the NN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1).astype('float32')
    y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1).astype('float32')
    print('Scaled data')

    
    if not os.path.exists(input_data_address):
        if not os.path.isdir('input_data/'):
          os.mkdir('input_data/')
        np.savetxt(input_data_address, DF.iloc[:, inputRange+outputRange])
        print('Saved training data in input_data as %s'%(run_name+'.txt'))
    return X_train, y_train, X_test, y_test, X_Scaler, y_Scaler, DF, input_data_reused


X_train, y_train, X_test, y_test, X_Scaler, y_Scaler, DF, input_data_reused = load_data(input_data_reused)
if input_data_reused:
    LCC_Var = DF.shape[1]-2
    CO2_Var = DF.shape[1]-1
lowest_training_LCC = np.min(DF[LCC_Var])
lowest_training_CO2 = np.min(DF[CO2_Var])


'''

##############################################################################
##############################################################################
##############################################################################

'''

# Parse the input arguments
if sys.argv[6] == '1':
    gen_architecture = [2,4,2]
    disc_architecture = [4,2,2]
    run_parameters = [2, 2]
    learningRate = 0.001
    latent_dim = 1
    testmode=True
    alpha = 0.2
    dropout = 0.4
    momentum = 0.8
    beta1 = 0.5
    beta2 = 0.999

else:
    architectures = parse_list_argv(sys.argv[2], 'int')
    gen_architecture, disc_architecture = architectures[:3], architectures[3:]
    
    run_parameters = parse_list_argv(sys.argv[3], 'int')
    learningRate = float(sys.argv[4])
    latent_dim = int(sys.argv[5])
    testmode = False
    alpha = float(sys.argv[7])
    dropout = float(sys.argv[8])
    momentum = float(sys.argv[9])  
    beta1 = float(sys.argv[10])
    beta2 = float(sys.argv[11])


now = datetime.now()

current_time_string = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
current_time = [now.year, now.month, now.day, now.hour,
                now.minute, now.second, now.microsecond]


# Go to the next line of the output file
with open(log_filename, "a") as f:
    f.write('\n')

# Log the information of the run
append_list_to_log(current_time)
append_list_to_log(gen_architecture)
append_list_to_log(disc_architecture)
append_list_to_log(run_parameters)
append_list_to_log([learningRate])
append_list_to_log([latent_dim])
append_list_to_log([alpha])
append_list_to_log([dropout])
append_list_to_log([momentum])
append_list_to_log([beta1])
append_list_to_log([beta2])


'''

##############################################################################
##############################################################################
##############################################################################

'''


### 1.2.2 Create and train the CGAN on the data


class CGAN():
    def __init__(self, X_train, y_train, X_test, y_test, X_Scaler, y_Scaler, latent_dim=3):
        # Input shape
        # self.img_rows = 1
        self.img_cols = len(X_train[0]) # Was 11 before Nov 5, 2020
        self.label_size = len(y_train[0])
        self.channels = 1
        self.img_shape = (self.img_cols, self.channels) #(self.img_rows, self.img_cols, self.channels)
        # self.num_classes = 10
        self.latent_dim = latent_dim # Was 22


        optimizer = Adam(learning_rate = learningRate, beta_1 = beta1, beta_2 = beta2) # Adam(0.0002, 0.5)


        self.X_train, self.y_train, self.X_Scaler, self.y_Scaler = \
            X_train, y_train, X_Scaler, y_Scaler



        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.label_size,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)



    def build_discriminator(self):
        # Best architecture: 1. 128-64-32 2. 256-256-128, 3. 512-256-128; 4. 512-512-512

        model = Sequential()
        
        model.add(Dense(disc_architecture[0], input_dim=np.prod(self.img_shape)+self.label_size))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dense(disc_architecture[1]))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout))
        model.add(Dense(disc_architecture[2]))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(self.label_size,), dtype='float32')

        # label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        # model_input = multiply([flat_img, label_embedding])
        model_input = Concatenate(axis=1)([flat_img, label])

        validity = model(model_input)

        return Model([img, label], validity)



    def build_generator(self):
      # Best architecture so far:
      # 1. 64-128-64; 2. 128-256-128; 3. 256 - 512 - 256

        model = Sequential()

        model.add(Dense(gen_architecture[0], input_dim=self.latent_dim+self.label_size))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dense(gen_architecture[1]))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dense(gen_architecture[2]))
        model.add(LeakyReLU(alpha=alpha))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        # model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.label_size,), dtype='float32')
        # label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        # model_input = multiply([noise, label_embedding])
        model_input = Concatenate(axis=1)([noise, label])
        img = model(model_input)

        return Model([noise, label], img)

    
  


    def train(self, epochs=1000, batch_size=128, sample_interval=200, runningAvgDur=10):
        # Load the dataset
        X_train, y_train =\
            self.X_train, self.y_train#, self.X_Scaler, self.y_Scaler
        # X_train, y_train, X_test, y_test, X_Scaler, y_Scaler = self.load_data()


        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        d_losses_real = []
        d_accs_real = []
        d_losses_fake = []
        d_accs_fake = []

        g_losses = []

        startOfDur = -int(runningAvgDur)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels

            sampled_indx = np.random.choice(y_train.shape[0], size=batch_size)  #np.random.randint(0, 10, batch_size).reshape(-1, 1)
            sampled_labels = y_train[sampled_indx]

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)


            # Register the progress
            d_losses_real.append(d_loss_real[0])
            d_accs_real.append(100*d_loss_real[1])

            d_losses_fake.append(d_loss_fake[0])
            d_accs_fake.append(100*d_loss_fake[1])

            g_losses.append(g_loss)

            



            # If at save interval => save generated image samples
            if (epoch > 0):
              if (epoch % sample_interval == 0) or (epoch == epochs - 1): #  and epoch >= 3*epochs/4
                self.sample_images(epoch)
                print("%d [D loss_real: %.3f, acc_real: %.2f%%] [D loss_fake: %.3f, acc_fake: %.2f%%] [G loss: %.3f]"
                  % (epoch, np.average(d_losses_real[startOfDur:]),
                     np.average(d_accs_real[startOfDur:]),
                     np.average(d_losses_fake[startOfDur:]),
                     np.average(d_accs_fake[startOfDur:]),
                     np.average(g_losses[startOfDur:])))
        

        self.plot_results(d_losses_real, d_losses_fake, g_losses, d_accs_real,
                          d_accs_fake, runningAvgDur, run_name, epochs)


        # Save the trained models
        # if not os.path.isdir('saved_model/'):
          # os.mkdir('saved_model/')
        # self.generator.save('saved_model/'+run_name+'_generator_model.tf')
        # self.discriminator.save('saved_model/'+run_name+'_discriminator_model.tf')
        # self.combined.save('saved_model/'+run_name+'_combined_model.tf')



    def sample_images(self, epoch):                        
        # Generate the random samples [i: LCC, j: GHG]
        [[min_observed_LCC, min_observed_GHG]] =\
            self.y_Scaler.inverse_transform([[-1,-1]])
            
        [[scaled_50_percent_better_LCC, scaled_50_percent_better_GHG]]  =\
            self.y_Scaler.transform([[0.5*min_observed_LCC, 0.5*min_observed_GHG]])
        
        if testmode:
            num1 = 1 # See below
            num2 = 1 # See below
        else:
            num1 = 10 # See below
            num2 = 10 # See below
            
        LCC_improving_range = np.linspace(-1, scaled_50_percent_better_LCC, num=num1)
        GHG_improving_range = np.linspace(-1, scaled_50_percent_better_GHG, num=num1)
        whole_range = np.linspace(-1, 1, num=num2)
        
        sampled_labels = []
        for i in LCC_improving_range: # Targetting better LCC and GHG 
            for j in GHG_improving_range:
                sampled_labels.append([i,j])
        
        for i in LCC_improving_range: # Targetting better LCC
            for j in whole_range:
                sampled_labels.append([i,j])
                
        for i in whole_range: # Targetting better GHG
            for j in GHG_improving_range:
                sampled_labels.append([i,j])


        r, c = len(sampled_labels), 1

        noise = np.random.normal(0, 1, (r * c, self.latent_dim))



        gen_imgs = self.generator.predict([noise, np.array(sampled_labels)])
        scaled_gen_imgs = self.X_Scaler.inverse_transform(gen_imgs)

        if not os.path.isdir('images/'):
          os.mkdir('images/')
        np.savetxt("images/"+current_time_string+run_name+"_%d.txt" % epoch, scaled_gen_imgs)



        
    def plot_results(self, d_losses_real, d_losses_fake, g_losses, d_accs_real,
                     d_accs_fake, runningAvgDur, run_name, epochs):
      # Plot the result of the model
      plt.style.use('ggplot')
      
      plt.figure()
      plt.plot(range(epochs), d_losses_real, color='red', alpha=0.1, label='Disc Loss Real')
      d_losses_real_avg = self.moving_average(d_losses_real, runningAvgDur)
      plt.plot(range(epochs), d_losses_real_avg, color='red', label='Disc Loss Real_Avg')

      plt.plot(range(epochs), d_losses_fake, color='green', alpha=0.1, label='Disc Loss Fake')
      d_losses_fake_avg = self.moving_average(d_losses_fake, runningAvgDur)
      plt.plot(range(epochs), d_losses_fake_avg, color='green', label='Disc Loss Fake_Avg')

      plt.plot(range(epochs), g_losses, color='blue', alpha=0.1, label='Gen Loss')
      g_losses_avg = self.moving_average(g_losses, runningAvgDur)
      plt.plot(range(epochs), g_losses_avg, color='blue', label='Gen Loss_Avg')

      plt.title('Discriminator & Generator Loss vs Iteration Number')
      plt.xlabel('Iteration #')
      plt.ylabel('Loss')
      plt.legend()
      plt.savefig('images/'+run_name+'_Disc Loss vs Gen Loss_'+current_time_string+'.png', bbox_inches='tight', dpi=400)
      # plt.show()

      plt.figure()
      plt.plot(range(epochs), d_accs_real, color='red', alpha=0.1, label='Disc Acc Real')
      d_accs_real_avg = self.moving_average(d_accs_real, runningAvgDur)
      plt.plot(range(epochs), d_accs_real_avg, color='red', label='Disc Acc Real_Avg')

      plt.plot(range(epochs), d_accs_fake, color='green', alpha=0.1, label='Disc Acc Fake')
      d_accs_fake_avg = self.moving_average(d_accs_fake, runningAvgDur)
      plt.plot(range(epochs), d_accs_fake_avg, color='green', label='Disc Acc Fake_Avg')

      plt.title('Discriminator Accuracy vs Iteration Number')
      plt.xlabel('Iteration #')
      plt.ylabel('Accuracy \%')
      plt.legend()
      plt.savefig('images/'+run_name+'_Disc Acc_'+current_time_string+'.png', bbox_inches='tight', dpi=400)
      # plt.show()
      
      plt.close('all')


    def moving_average(self, a, n=3):
        # Calculate the moving average of array a with window_size n
        if n == 1:
            return a
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        ret[:n-1] /= np.arange(1,n)
        ret[n - 1:] /= n
        return ret
    
    
    
    
# Train the CGAN and generate outputs
# latent_dim = 3
cgan = CGAN(X_train, y_train, X_test, y_test, X_Scaler, y_Scaler, latent_dim=latent_dim)
# Long runs
if testmode:
    runningAvgDur = 1
else:
    runningAvgDur = 10
cgan.train(epochs=run_parameters[0], batch_size=run_parameters[1],
           sample_interval=round(run_parameters[0]/20), runningAvgDur=runningAvgDur)

                



# Copy all the images to Data directory to be processed by the evaluation function
from shutil import copyfile

if not os.path.isdir('Data/'):
  os.mkdir('Data/')

for filename in os.listdir('images/'):
    if run_name in filename and '.txt' in filename and current_time_string in filename: # i.e. if the file is a results file for this run_name
      copyfile('images/'+filename, 'Data/'+filename)
print('Copied all generated results from images to Data directory')




'''

##############################################################################
##############################################################################
##############################################################################

'''


# 2 Evaluate the generated solutions
# Constants for plotting the filtered plots
LCC_Cutoff = 100 # k$/m2
CO2_Cutoff = 10 # T-CO2/m2



# Constants for evaluating the solutions
len_of_indiv = len(X_train[0])



# Load the libraries

if os.name == 'posix': # To run on Sherlock
    Ch3_Dir = '/home/users/pouyar/GAN_on_GA/Simulation'
else:
    Ch3_Dir = './Simulation'
sys.path.append(Ch3_Dir)

os.chdir(Ch3_Dir)
from Main import SupplyandDemandOptimization as analyze, Low_Seq, High_Seq

if os.name == 'posix': # To run on Sherlock
    os.chdir('/home/users/pouyar/GAN_on_GA')
else:
    os.chdir('..')


# from Plots_Paper_One import DF_Filter

# from tqdm import tqdm


'''

##############################################################################
##############################################################################
##############################################################################

'''

## 2.2 Helper functions

## Helper function for loading the filenames of the results
def load_filenames(experiment):
  # TO DO: TODO use the results from images/<run_name>_*.txt
  fileNames = []
  for filename in os.listdir('images/'):
    if run_name in filename and '.txt' in filename and current_time_string in filename: # i.e. if the file is a results file for this run_name
      fileNames.append(filename)
  return fileNames


  
'''

##############################################################################
##############################################################################
##############################################################################

'''


## 2.3 Main
# DECLARE THE EXPERIMENT TYPE
plot_only = False # If you want no evaluation and simply plotting the results, set this to True; no plots are made when this is False
stats_and_plot = True # If you want statistical info to be printed out while plot_only is True, set this to True
aggregate_results = False # If you want the results of the long and short runs to get combined, set this to be True
verbose = True # If you want full outputs to be printed out, set this to True


plt.close('all')
# Load the proper set of filenames based on the experiment
fileNames = load_filenames(experiment)


# Read in the data generated by GAN or load the already processed dataset
if not plot_only:
    if not os.path.exists('Data/rebuilt_'+current_time_string+'.txt'):
        # If more than one filename provided, combine them and save them as one
        loadedData = np.round(np.loadtxt('Data/'+fileNames[0]))
        if len(fileNames) > 1:
            for name in fileNames[1:]:
                loadedData = np.append(loadedData, np.round(np.loadtxt('Data/'+name)), axis=0)
            
        np.savetxt('Data/rebuilt_'+current_time_string+'.txt', loadedData)
        print('Saved rebuilt data generated by GAN into %s'%('Data/rebuilt_'+current_time_string+'.txt'))
    else:
        loadedData = np.loadtxt('Data/rebuilt_'+current_time_string+'.txt')
        print('Loaded rebuilt data generated by GAN from %s'%('Data/rebuilt_'+current_time_string+'.txt'))





# Evaluate the individuals or load the already-evaluated individuals
if not aggregate_results and not os.path.exists('Data/resultsTotal_'+current_time_string+'.txt'):
    # Evaluate the individuals
    resultsTotal = []
    correctedIndivs = 0 # Number of individuals corrected (meaning they were outside the allowed boundary)
    ignoredIndivs = 0 # Number of ignored individuals

    for indiv in loadedData:
        indiv = list(indiv)
        # Fit the generated individual inside the High_Seq - Low_Seq range
        oldIndiv = indiv.copy()
        
        # Check the validity of the DNA and round to the nearest valid number
        indiv = list(np.minimum(np.maximum(indiv, Low_Seq), High_Seq))
        
        # Check if there are any buildings in the development or nan exists in the decision variables
        if np.sum(indiv[:4]) == 0 or np.nan in indiv:
            ignoredIndivs += 1
            continue
        
        # Register if the individual has been corrected
        if indiv != oldIndiv:
            correctedIndivs += 1

        results = analyze(indiv)

        # Skip trivial cases
        if float('inf') in results[0] or -float('inf') in results[0]: 
            ignoredIndivs += 1
            continue

        [LCC_per_GFA, CO2_per_GFA] = results[0]
        if LCC_per_GFA != float('inf'):
            indiv.extend(results[0])
            resultsTotal.append(indiv)

    print('\n\n++++++++++++++++++++++++++\nNumber of corrected individuals: %d out of %d'%(correctedIndivs, len(loadedData)))
    print('Number of ignored individuals: %d out of %d'%(ignoredIndivs, len(loadedData)))

    np.savetxt('Data/resultsTotal_'+current_time_string+'.txt', resultsTotal)
    print('Saved points generated by GAN and their associated results into %s'%('Data/resultsTotal_'+current_time_string+'.txt'))

else:
    if not aggregate_results:
        resultsTotal = np.loadtxt('Data/resultsTotal_'+current_time_string+'.txt')
        print('Loaded points generated by GAN from %s'%('Data/resultsTotal_'+current_time_string+'.txt'))
    else:
        pass
        # resultsTotal = load_combined_results(experiment)

resultsTotal = np.array(resultsTotal)


'''

##############################################################################
##############################################################################
##############################################################################

'''
    
## 2.4 Plot the results [deactivated] and calculate the improvements in the generated solutions compared to the original solutions
   
DFName = run_name

plt.style.use('ggplot')
colors_rb = {DFName:'r'}

#############################################

# =============================================================================
#                         if plot_only or stats_and_plot:
#                             if verbose: print('\nPlotting!\n')
#                         
#                         
#                             # LCC vs CO2
#                             if verbose: print('Plotting LCC vs CO2')
#                             plt.figure(figsize=(10,5))
#                             plt.scatter(x=DF[LCC_Var]/10**3,y=DF[CO2_Var], label=DFName, s=100, alpha=1, c=colors_rb[DFName], marker='x')
#                             plt.scatter(x=resultsTotal[:,len_of_indiv+0]/10**3, y=resultsTotal[:,len_of_indiv+1], label='C-GAN', s=150, alpha=1, c='blue', marker='+')
#                             
#                             plt.xlabel(r'LCC (k\$/$m^2$)')
#                             plt.ylabel(r'GHG (T-$CO_{2e}$/$m^2$)')
#                             plt.legend()
#                             
#                             if not os.path.isdir('Figures/'):
#                               os.mkdir('Figures/')
#                         
#                             plt.savefig('Figures/All_CO2_vs_LCC'+current_time_string+'.png', dpi=400, bbox_inches='tight')
# =============================================================================
    




if not plot_only or stats_and_plot:
    # Check the improvement in targeted objective function compared to the training data
    LCC_Var_Gen = len_of_indiv+0
    CO2_Var_Gen = len_of_indiv+1
    
    lowest_generated = np.min(resultsTotal[:,LCC_Var_Gen])
    print('lowest LCC in generated individuals: %.2f'%lowest_generated)
    append_list_to_log([lowest_generated])
    print('lowest LCC in training individuals: %.2f'%lowest_training_LCC)
    append_list_to_log([lowest_training_LCC])
    print('--> Percent improvement: %.1f%%'%((lowest_training_LCC-lowest_generated)/lowest_training_LCC*100))
    
    lowest_generated = np.min(resultsTotal[:,CO2_Var_Gen])
    print('Lowest CO2 in generated data: %.4f'%lowest_generated)
    append_list_to_log([lowest_generated])
    print('Lowest CO2 in training data: %.4f'%lowest_training_CO2)
    append_list_to_log([lowest_training_CO2])
    print('--> Percent improvement: %.1f%%'%((lowest_training_CO2-lowest_generated)/lowest_training_CO2*100))
        
    
    
    
    
    ## 2.4 Calculate the changes in hypervolume
    
    # Constants
    ref_point = [1.0, 1.0] # Reference point for calculating the hypervolume
    np.random.seed(42)
    
    # Calculate the hypervolume
    resultsTotal = pd.DataFrame(resultsTotal)
    ## modify the input parameters to make them suitable for calculating the HV w.r.t. the reference point (1,1,1)
    maxLCC = max(np.max(DF[LCC_Var]), np.max(resultsTotal[LCC_Var_Gen]))
    maxCO2 = max(np.max(DF[CO2_Var]), np.max(resultsTotal[CO2_Var_Gen]))
    
    minLCC = min(np.min(DF[LCC_Var]), np.min(resultsTotal[LCC_Var_Gen]))
    minCO2 = min(np.min(DF[CO2_Var]), np.min(resultsTotal[CO2_Var_Gen]))
    
    ## Normalize the generated and training data to 0 to 1 range
    def normalize_data(df, column, minValue, maxValue):
        df[column] = (df[column] - minValue)/(maxValue - minValue)
    
    newDF = DF[[LCC_Var, CO2_Var]]
    normalize_data(newDF, LCC_Var, minLCC, maxLCC)
    normalize_data(newDF, CO2_Var, minCO2, maxCO2)
    
    normalize_data(resultsTotal, LCC_Var_Gen, minLCC, maxLCC)
    normalize_data(resultsTotal, CO2_Var_Gen, minCO2, maxCO2)
    
    ## Calculate the hv
    from pymoo.factory import get_performance_indicator
    hv = get_performance_indicator("hv", ref_point=np.array(ref_point))
    
    ## GIVES MEMORY ERROR IF USED DIRECTLY ##
    array1 = np.array(newDF[[LCC_Var, CO2_Var]])
    
    array2 = np.array(resultsTotal[[LCC_Var_Gen, CO2_Var_Gen]])
    generatedArea = hv.calc(array2)
    
    originalAreas = []
    # prevArr = None
    Num_Samplings = np.minimum(100, int(len(array1)/len(array2))+1)
    for i in range(Num_Samplings):
        choices = np.random.randint(low=0, high=len(array1), size=len(array2))
        array1_2 = np.array(newDF[[LCC_Var, CO2_Var]])[choices, :]
        originalAreas.append(hv.calc(array1_2))
    
    meanOriginalArea = np.mean(originalAreas)
    maxOriginalArea = np.max(originalAreas)
    
    print('Generated hv:%.2e'%generatedArea)
    append_list_to_log([generatedArea])
    print('Maximum original hv:%.2e'%maxOriginalArea)
    append_list_to_log([maxOriginalArea])
    print('Mean original hv:%.2e'%meanOriginalArea)
    append_list_to_log([meanOriginalArea])
    if generatedArea > meanOriginalArea:
        print('Generated solutions have on average a hv %.2f%% larger than the original solutions'%((generatedArea - meanOriginalArea)/meanOriginalArea*100))
    else:
        print('Original solutions have on average a hv %.2f%% larger than the generated solutions'%((meanOriginalArea - generatedArea)/meanOriginalArea*100))
        
    if generatedArea > maxOriginalArea:
        print('Generated solutions have a hv %.2f%% larger than the best of original solutions'%((generatedArea - maxOriginalArea)/maxOriginalArea*100))
    else:
        print('Original solutions have at best a hv %.2f%% larger than the generated solutions'%((maxOriginalArea - generatedArea)/maxOriginalArea*100))
    
    
# =============================================================================
#                         if plot_only or stats_and_plot:
#                             # LCC vs CO2 (Filtered below the cutoff thresholds)
#                             ## Filter dataframes based on max values of CO2 and LCC
#                             if verbose: print('++++\nFiltering DF and resultsTotal based on cutoffs (LCC_Cutoff: %d, CO2_Cutoff: %d)'%(LCC_Cutoff, CO2_Cutoff))
#                             DF = DF[DF[LCC_Var]/10**3 <= LCC_Cutoff]
#                             DF = DF[DF[CO2_Var] <= CO2_Cutoff] # Observation: with a less than $500k/m2 of LCC, no designs have CO2/m2 higher than 5 T-CO2/m2
#                             resultsTotal = resultsTotal[resultsTotal[:,len_of_indiv+0]/10**3 <= LCC_Cutoff]
#                             resultsTotal = resultsTotal[resultsTotal[:,len_of_indiv+1] <= CO2_Cutoff]
#                             
#                             
#                             
#                             if verbose: print('Plotting LCC vs CO2')
#                             plt.figure(figsize=(10,5))
#                             plt.scatter(x=DF[LCC_Var]/10**3,y=DF[CO2_Var], label=DFName, s=100, alpha=1, c=colors_rb[DFName], marker='x')
#                             plt.scatter(x=resultsTotal[:,len_of_indiv+0]/10**3, y=resultsTotal[:,len_of_indiv+1], label='C-GAN', s=150, alpha=1, c='blue', marker='+')
#                             
#                             plt.xlabel(r'LCC (k\$/$m^2$)')
#                             plt.ylabel(r'GHG (T-$CO_{2e}$/$m^2$)')
#                             plt.legend()
#                             if not os.path.isdir('Figures/'):
#                               os.mkdir('Figures/')
#                             plt.savefig('Figures/CO2_vs_LCC'+current_time_string+'.png', dpi=400, bbox_inches='tight')
# =============================================================================
