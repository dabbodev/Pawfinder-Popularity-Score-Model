wrapper = keras.Sequential([
    model1,
    ScoreRenderer(),
    layers.Dense(1, use_bias=False, input_shape=(None,5), kernel_constraint=tf.keras.constraints.UnitNorm(axis=0))
])

wrapper.compile(loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE), optimizer=tf.keras.optimizers.Adadelta(learning_rate=0.1, rho=(58/59)), metrics = ['RootMeanSquaredError'])

lastLoss = 999
overlap=4
for x in range(batches + overlap - 1):
    if (x >= batches):
        x = batches - 1
    for z in range(0, x):
        print(f'Starting Batch {z+1} out of {batches - 1}')
        wrapper, lastLoss = train_batch(z, wrapper, batch_length, 4, lastLoss, f=-1, phase=2, ls=4)