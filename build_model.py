

from __init__ import *


class BuildModel:
	def __init__(self):
		pass


datadir = "/home/ovi/PROJECTS_YEAR_4/SMART_TECH/SmartTechCA2Data/track_2/one_lap/"
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(datadir, 'driving_log.csv'), names = columns)
# print(data.head)

def path_leaf(path):
  head, tail = ntpath.split(path)


  return tail

data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)





num_bins = 15
hist, bins = np.histogram(data['steering'], num_bins)
# print(bins)
# plt.hist(data['steering'], bins=num_bins, alpha=0.5, color='blue', edgecolor='black')

# # Add labels and title
# plt.xlabel('Steering Angle')
# plt.ylabel('Frequency')
# plt.title('Steering Angle Histogram')

# # Show the plot
# plt.show()
samples_per_bin = 175

remove_list = []
for j in range(num_bins):
  list_ = []
  for i in range(len(data['steering'])):
    if bins[j] <= data['steering'][i] <= bins[j+1]:
      list_.append(i)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bin:]
  remove_list.extend(list_)

# print("Removed: ", len(remove_list))
data.drop(data.index[remove_list], inplace = True)
# print("Remaining: ", len(data))

num_bins = 15
hist, bins = np.histogram(data['steering'], num_bins)
# print(bins)
# plt.hist(data['steering'], bins=num_bins, alpha=0.5, color='blue', edgecolor='black')

# # Add labels and title
# plt.xlabel('Steering Angle')
# plt.ylabel('Frequency')
# plt.title('Steering Angle Histogram')

# # Show the plot
# plt.show()


def load_img_steering(datadir, df):
  image_path = []
  steering = []
  offset = 0.3333
  for i in range(len(df)):
    indexed_data = data.iloc[i]
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    #center image path and steering value
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[3]))
    #left image path and steering value
    image_path.append(os.path.join(datadir, left.strip()))
    steering.append(float(indexed_data[3])+offset)
        #left image path and steering value
    image_path.append(os.path.join(datadir, right.strip()))
    steering.append(float(indexed_data[3])-offset)
    
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)

  return image_paths, steerings


# load_img_steering(datadir+'/IMG', data)



def pan(img):
  pan_func=iaa.Affine(translate_percent={"x":(-0.1,0.1),"y":(-0.1,0.1)})
  pan_image=pan_func.augment_image(img)
  return pan_image

def zoom(img):
  zoom_func=iaa.Affine(scale=(1,1.3))
  z_img=zoom_func.augment_image(img)
  return z_img

def img_random_brightness(img):
  bright_func=iaa.Multiply((0.2,1.2))
  bright_img=bright_func.augment_image(img).astype("uint8")
  return bright_img

def img_random_flip(img,steering_angle):
  flipped_img=cv2.flip(img,1)
  steering_angle=-steering_angle
  return flipped_img,steering_angle

def batch_generator(image_paths,steering_angle,batch_size,is_training):
  while True:
    batch_img=[]
    batch_steering=[]
    for i in range(batch_size):
      random_index=random.randint(0,len(image_paths)-1)
      if is_training:
        im, steering=random_augment(image_paths[random_index],steering_angle[random_index])
      else:
        im=mpimg.imread(image_paths[random_index])
        steering=steering_angle[random_index]
      im=preprocess_img_no_imread(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield np.asarray(batch_img),np.asarray(batch_steering)


def preprocess_img(img):
  img = mpimg.imread(img)
  img = img[60:135, :, :]
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img, (3,3), 0)
  img = cv2.resize(img, (200, 66))
  img = img/255
  return img

def preprocess_img_no_imread(img):
  # img = mpimg.imread(img)
  img = img[60:135, :, :]
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img, (3,3), 0)
  img = cv2.resize(img, (200, 66))
  img = img/255
  return img

#all image processing methods work except flip
# image =image_paths[100]
# original_image=mpimg.imread(image)
# preprocessed_image=zoom(original_image)
# fig,axes=plt.subplots(1,2,figsize=(15,10))
# fig.tight_layout()
# axes[0].imshow(original_image)
# axes[0].set_title("Original Image")
# axes[1].imshow(preprocessed_image)
# axes[1].set_title("Preprocessed Image")
# plt.show()


def random_augment(image_to_augment, steering_angle):
    augment_image = mpimg.imread(image_to_augment)
    if np.random.rand() < 0.5:
        augment_image = zoom(augment_image)
    if np.random.rand() < 0.5:
        augment_image = pan(augment_image)
    if np.random.rand() < 0.5:
        augment_image = img_random_brightness(augment_image)
    if np.random.rand() < 0.5:
        augment_image, steering_angle = img_random_flip(augment_image, steering_angle)
    return augment_image, steering_angle





# image_paths, steerings = load_img_steering(datadir+'/IMG', data)
# X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)

# print(f"Traing samples {len(X_train)}, validation samples {len(X_valid)}")
# X_train = np.array(list(map(preprocess_img, X_train)))
# X_valid = np.array(list(map(preprocess_img, X_valid)))




image_paths, steerings = load_img_steering(datadir+'/IMG', data)
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)

print(f"Traing samples {len(X_train)}, validation samples {len(X_valid)}")

# x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
# x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))
# fig, axs = plt.subplots(1, 2, figsize=(15, 10))
# fig.tight_layout()
# axs[0].imshow(x_train_gen[0])
# axs[0].set_title("Training Image")
# axs[1].imshow(x_valid_gen[0])
# axs[1].set_title("Validation Image")
# plt.show()






# https://arxiv.org/pdf/1604.07316v1.pdf
def nvidia_model():
  model = Sequential()
  model.add(Convolution2D(24, kernel_size=(5,5), strides=(2,2), input_shape=(66, 200, 3), activation='elu'))
  model.add(Convolution2D(36, kernel_size=(5,5), strides=(2,2), activation='elu'))
  model.add(Convolution2D(48, kernel_size=(5,5), strides=(2,2), activation='elu'))
  model.add(Convolution2D(64, kernel_size=(3,3), activation='elu'))
  model.add(Convolution2D(64, kernel_size=(3,3), activation='elu'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(Dense(100, activation = 'elu'))
  model.add(Dropout(0.5))
  model.add(Dense(50, activation = 'elu'))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation = 'elu'))
  model.add(Dropout(0.5))
  model.add(Dense(1))

  optimizer = Adam(learning_rate = 0.0001)
  model.compile(loss='mse', optimizer = optimizer)
  return model

model = nvidia_model()
print(model.summary())

# history = model.fit(X_train, y_train, epochs=40, 
#   validation_data = (X_valid, y_valid), batch_size=100, verbose=1, shuffle = 2)

# model.save('./model/model_1_track_2_onelap.h5')

 

history = model.fit(batch_generator(X_train, y_train,100,1),steps_per_epoch=100, epochs=20, 
  validation_data = batch_generator(X_valid, y_valid,100,0), validation_steps=200, verbose=1, shuffle = 2)

model.save('./model/model_1_track_2_onelap.h5')