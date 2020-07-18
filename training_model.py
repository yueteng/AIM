import buliding_model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
pos_dataset = "Parasitized"  #Folder for positive sample images
neg_dataset = "Uninfected"   #Folder for negative sample images 
saving_dataset = "data"      #Converts the image to.npy and stores it in this folder
model_name = "the_model_name.h5"  # The model name
model = ["VGG", "ResNet", "Alex"] # Network structure of encoder
buliding_model.create_data.create_dataset(pos_dataset, neg_dataset, saving_dataset)
buliding_model.train_model.train_model(saving_dataset, model_name, model[0])#
buliding_model.train_model.test_model(saving_dataset, model_name)
