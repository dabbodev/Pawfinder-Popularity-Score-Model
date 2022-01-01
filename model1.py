# IMPORT LAYERS
import SoftBinaryCap
import binaryFeatureDetector

# INITIALIZE VALIDATION DATA SET
val_x = []
val_y = []
val_data = assemble_batch(batches-1, batch_length)

for feature, labels in val_data:
    val_x.append(feature)
    val_y.append(labels[1:])

del val_data    
        
for x in range(len(val_y)):
    val_y[x] = val_y[x].to_list()
    for y in range(len(val_y[x])):
        if (y == len(val_y[x]) - 1):
            encoded = encodescore(val_y[x][y])
            val_y[x][y] = encoded[0]
            val_y[x].append(encoded[1])
            val_y[x].append(encoded[2])
        val_y[x][y] = float(val_y[x][y])
    val_y[x] = np.array(val_y[x])
        
val_x = np.array(val_x) / 255
val_x = preprocessdata(val_x)
    
val_y = np.asarray(val_y).astype('float32')


# INITIALIZE RES50 MODEL FROM IMAGENET
res_model = keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=keras.Input(shape=(img_size, img_size, 3)))

# MAKE TOP UNTRAINABLE TO MAINTAIN WEIGHTS
for layer in res_model.layers[:143]:
    layer.trainable = False

# CREATE PARALLEL CONVOLUTION STREAMS
ip_shape = (204800,) # OUTPUT FROM RES50
inp = keras.Input(shape=ip_shape)
convs = []
for x in range(15): #12 BINARY FEATURES + 3 BINARY SCORE ENCODERS
    convs.append(binaryFeatureDetector()(inp))

out = layers.Concatenate()(convs)
conv_model = keras.Model(inputs=inp, outputs=out)

model1 = keras.Sequential([
    res_model,
    layers.Flatten(),
    conv_model
])

# COMPILE AND TRAIN
model1.compile(loss=tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE), optimizer=tf.keras.optimizers.Adadelta(learning_rate=1e-7, rho=(58/59)), metrics=['RootMeanSquaredError', 'binary_accuracy'])

lastLoss = 999
overlap=4
for x in range(batches + overlap - 1):
    if (x >= batches):
        x = batches - 1
    for z in range(0, x):
        print(f'Starting Batch {z+1} out of {batches - 1}')
        model1, lastLoss = train_batch(z, model1, batch_length, 4, lastLoss, q=100., ls=4)

# MAKE BINARY FEATURES UNTRAINABLE
for layer in model1.layers[2].layers[1:13]:
    layer.trainable = False

# MAKE RES50 UNTRAINABLE
for layer in model1.layers[0].layers:
    layer.trainable = False