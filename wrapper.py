from model1 import model1
import ScoreRenderer

# CREATE VALIDATION DATA SET
val_x = []
val_y = []

val_data = assemble_batch(batches-1, batch_length)

for feature, labels in val_data:
    val_x.append(feature)
    val_y.append(labels[-1:])

del val_data    
        
for x in range(len(val_y)):
    val_y[x] = val_y[x].to_list()
    for y in range(len(val_y[x])):
        if (y == len(val_y[x]) - 1):
            val_y[x][y] = float(val_y[x][y])
        val_y[x][y] = float(val_y[x][y])
    val_y[x] = np.array(val_y[x])
        
val_x = np.array(val_x) / 255
val_x = preprocessdata(val_x)
    
val_y = np.asarray(val_y).astype('float32')

# WRAP TOP MODEL WITH RENDERER
wrapper = keras.Sequential([
    model1,
    ScoreRenderer(),
    layers.Dense(1, use_bias=False, input_shape=(None,5), kernel_constraint=tf.keras.constraints.UnitNorm(axis=0))
])

#COMPILE AND TRAIN
wrapper.compile(loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE), optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1, rho=(58/59)), metrics = ['RootMeanSquaredError'])

lastLoss = 999
overlap=4
for x in range(batches + overlap - 1):
    if (x >= batches):
        x = batches - 1
    for z in range(0, x):
        print(f'Starting Batch {z+1} out of {batches - 1}')
        wrapper, lastLoss = train_batch(z, wrapper, batch_length, 4, lastLoss, f=-1, phase=2, ls=4)