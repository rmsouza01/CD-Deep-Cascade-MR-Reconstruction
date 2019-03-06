import numpy as np
import keras
import os

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs,under_masks,norm,batch_size=8, dim=(218,170), n_channels=2,
                 shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        #self.data_path = data_path
        self.under_masks = under_masks 
        self.norm = norm
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

    # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0],self.dim[1], self.n_channels))
        y1 = np.empty((self.batch_size, self.dim[0],self.dim[1], 2))
        y2 = np.empty((self.batch_size, self.dim[0],self.dim[1], 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            aux = np.load(ID)/self.norm
            X[i,:,:,0] = aux.real
            X[i,:,:,1] = aux.imag
        aux2 = np.fft.ifft2(X[i,:,:,0]+1j*X[i,:,:,1])
        y1[:,:,:,0] = aux2.real
        y1[:,:,:,0] = aux2.imag
        y2[:,:,:,0] = np.abs(aux2)
        indexes = np.random.choice(np.arange(self.under_masks.shape[0], dtype=int), self.batch_size, replace = True)
        mask_X = self.under_masks[indexes] 
        X[mask_X] = 0 
        return [X,mask_X], [y1,y2]