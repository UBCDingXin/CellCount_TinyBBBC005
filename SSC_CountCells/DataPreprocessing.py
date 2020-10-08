from PIL import Image
import os
import numpy as np
import csv

# put train and test folders in the data folder
data_dir = '/home/xin/Working directory/Counting_Cells/SSC_CountCells/data'
#data_dir = '/media/qiong/icecream/SSC_CountCells/data'

#!!! resize ?!!!
resize_imgs = True


### read training labels ####
os.chdir(data_dir)#please change the directory to your working path
training_image_name = []
training_cell_count = []
training_blur_level = []
training_stain = []
with open('train_label.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader,None)
    for row in reader:
        training_image_name.append(row[0])
        training_cell_count.append(int(row[1]))
        training_blur_level.append(int(row[2]))
        training_stain.append(int(row[3]))
#convert to numpy
training_image_name = np.string_(training_image_name)
training_cell_count = np.array(training_cell_count,dtype = np.uint8)
training_blur_level = np.array(training_blur_level,dtype = np.uint8)
training_stain = np.array(training_stain,dtype = np.uint8)

### read test labels ####
os.chdir(data_dir)#please change the directory to your working path
testing_image_name = []
testing_blur_level = []
testing_stain = []
testing_cell_count = []
with open('test_label_new.csv', newline='') as f: 
    reader = csv.reader(f)
    next(reader,None)
    for row in reader:
        testing_image_name.append(row[0])
        testing_blur_level.append(row[1])
        testing_stain.append(row[2])
        testing_cell_count.append(int(row[3]))
#convert to numpy
testing_image_name = np.string_(testing_image_name)
testing_blur_level = np.array(testing_blur_level,dtype = np.uint8)
testing_stain = np.array(testing_stain,dtype = np.uint8)
testing_cell_count = np.array(testing_cell_count,dtype=np.uint8)

if resize_imgs:
    ### read training images ####
    os.chdir(data_dir + '/train/')#please change the directory to your working path
    IMGs_train = np.zeros((len(training_image_name), 1, 520, 696),dtype = np.uint8)
    for i in range(len(training_image_name)):
        im = Image.open(training_image_name[i])
        #im.show()
        IMGs_train[i][0] = np.array(im)
    
    ## read testing images ####        
    os.chdir(data_dir + '/test/')#please change the directory to your working path
    IMGs_test = np.zeros((len(testing_image_name), 1, 520, 696), dtype = np.uint8)
    for i in range(len(testing_image_name)):
        im = Image.open(testing_image_name[i])
        #im.show()
        IMGs_test[i][0] = np.array(im)
    
    #############################################################
    # create hdf5 file to store loaded images, features and outcomes via h5py
    #############################################################
    import h5py
    os.chdir(data_dir)
    # h5py_file = 'CellCount_dataset.h5'
    h5py_file = 'CellCount_resized_dataset.h5'
    with h5py.File(h5py_file, "w") as f:
        f.create_dataset('IMGs_train', data = IMGs_train)
        f.create_dataset('IMGs_Names_train', data = training_image_name)
        f.create_dataset('Blur_train', data = training_blur_level)
        f.create_dataset('Stain_train', data = training_stain)
        f.create_dataset('CellCount_train', data = training_cell_count)
        f.create_dataset('IMGs_test', data = IMGs_test)
        f.create_dataset('IMGs_Names_test', data = testing_image_name)
        f.create_dataset('Blur_test', data = testing_blur_level)
        f.create_dataset('Stain_test', data = testing_stain)
        f.create_dataset('CellCount_test', data = testing_cell_count)
        
    ##test hdf5 file
    #import os
    #data_dir = "/home/xin/ownCloud/Working directory/Counting_Cells/Data Files_Question1_SSC2019CaseStudy/"
    #os.chdir(data_dir)
    #import numpy as np
    #import h5py
    #h5py_file = 'CellCount_dataset.h5'
    #hf = h5py.File(h5py_file, 'r')
    #IMGs_train = hf['IMGs_train'].value
    #Names_train = hf['IMGs_Names_train'].value
    #Blur_train = hf['Blur_train'].value
    #Stain_train = hf['Stain_train'].value
    #CellCount_train = hf['CellCount_train'].value
    #IMGs_test = hf['IMGs_test'].value
    #Names_test = hf['IMGs_Names_test'].value
    #Blur_test = hf['Blur_test'].value
    #Stain_test = hf['Stain_test'].value
    #CellCount_test = hf['CellCount_test'].value
    #hf.close()
else:
    #############################################################
    # resize images and dump to h5 file
    #############################################################
    resize = [300,300]
    
    def resize_image(img_file_name, load_dir, save_dir, width=resize[0], height=resize[1]):
        # print(img_file_name)
        image = Image.open(os.path.join(load_dir,img_file_name))
        new_image = image.resize((width, height), resample=Image.LANCZOS)
        new_image.save(os.path.join(save_dir, img_file_name))
        
    source_folder = './train'
    target_folder = './train_resized'
    image_name = os.listdir(source_folder)
    if not os.path.exists(target_folder):
    	os.makedirs(target_folder)
    for name in image_name:
        if(name[0].isalpha()):
        	resize_image(name, source_folder, target_folder)
    
    source_folder = './test'
    target_folder = './test_resized'
    image_name = os.listdir(source_folder)
    if not os.path.exists(target_folder):
    	os.makedirs(target_folder)
    for name in image_name:
        if(name[0].isalpha()):
        	resize_image(name, source_folder, target_folder)
    
    ### read training images ####
    os.chdir(data_dir + '/train_resized/')#please change the directory to your working path
    IMGs_train_resized = np.zeros((len(training_image_name), 1, resize[0], resize[1]),dtype = np.uint8)
    for i in range(len(training_image_name)):
        im = Image.open(training_image_name[i])
        #im.show()
        IMGs_train_resized[i][0] = np.array(im)
    
    ## read testing images ####        
    os.chdir(data_dir + '/test_resized/')#please change the directory to your working path
    ##create a list (testing_list) containing pixel value of 1200 testing images
    IMGs_test_resized = np.zeros((len(testing_image_name), 1, resize[0], resize[1]), dtype = np.uint8)
    for i in range(len(testing_image_name)):
        im = Image.open(testing_image_name[i])
        #im.show()
        IMGs_test_resized[i][0] = np.array(im)
    
    
    #############################################################
    # create hdf5 file to store loaded images, features and outcomes via h5py
    #############################################################
    import h5py
    os.chdir(data_dir)
    h5py_file = 'CellCount_resized_dataset.h5'
    with h5py.File(h5py_file, "w") as f:
        f.create_dataset('IMGs_resized_train', data = IMGs_train_resized)
        f.create_dataset('IMGs_Names_train', data = training_image_name)
        f.create_dataset('Blur_train', data = training_blur_level)
        f.create_dataset('Stain_train', data = training_stain)
        f.create_dataset('CellCount_train', data = training_cell_count)
        f.create_dataset('IMGs_resized_test', data = IMGs_test_resized)
        f.create_dataset('IMGs_Names_test', data = testing_image_name)
        f.create_dataset('Blur_test', data = testing_blur_level)
        f.create_dataset('Stain_test', data = testing_stain)
        f.create_dataset('CellCount_test', data = testing_cell_count)









