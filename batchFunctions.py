def encodescore(X):
    out = [0, 0, 0]
    if (X < 25):
        return out
    elif (X < 50):
        out = [0, 0, 1]
        return out
    elif (X < 75):
        out = [0, 1, 1]
        return out
    else:
        out = [1, 1, 1]
        return out

def preprocessdata(X):
    X_p = keras.applications.resnet50.preprocess_input(X)
    return X_p
        

def assemble_batch(batch_num, batch_length):
    start_index = batch_num * batch_length
    total_length = scores.shape[0]
    data = []
    if (batch_num == batches - 1):
        if (total_length % batches > 0):
            batch_length = batch_length + (total_length % batches)
    end_index = start_index + batch_length
    for x in range(start_index, end_index):
        entry = scores.iloc[x]
        try:
            img_arr = cv2.imread(os.path.join(path + 'train', entry.Id + '.jpg'))[...,::-1] 
            resized_arr = cv2.resize(img_arr, (img_size, img_size)) 
            data.append([resized_arr, entry])
        except Exception as e:
            print(e)
    print(f'Batch {batch_num+1} assembled. Length: {batch_length}')
    return np.array(data, dtype=object)
    
def train_batch(z, model, batch_length, max_epochs, lastLoss=999, rMetric="loss", f=1, q=1.0, d=0.0, phase=1, ls=4): 
    global batchrates
    global current_batch
    global val_x
    global val_y
    train = assemble_batch(z, batch_length)
    current_batch = z
    x_train = []
    y_train = []
    
    if (phase == 1):     
        for feature, labels in train:
            x_train.append(feature)
            y_train.append(labels[f:])

        del train    
        
        for x in range(len(y_train)):
            y_train[x] = y_train[x].to_list()
            for y in range(len(y_train[x])):
                if (y == len(y_train[x]) - 1):
                    encoded = encodescore(y_train[x][y])
                    y_train[x][y] = encoded[0]
                    y_train[x].append(encoded[1])
                    y_train[x].append(encoded[2])
                y_train[x][y] = float(y_train[x][y])
            y_train[x] = np.array(y_train[x])
        
        x_train = np.array(x_train) / 255
        x_train = preprocessdata(x_train)
    
        y_train = np.asarray(y_train).astype('float32')
        
    elif (phase == 2):
        for feature, labels in train:
            x_train.append(feature)
            y_train.append(labels[-1:])
            
        del train 
            
        for x in range(len(y_train)):
            y_train[x] = y_train[x].to_list()
            for y in range(len(y_train[x])):
                y_train[x][y] = float(y_train[x][y])
            y_train[x] = np.array(y_train[x])
            
        x_train = np.array(x_train) / 255
        x_train = preprocessdata(x_train)
        y_train = np.asarray(y_train).astype('float32')
    
    history = model.fit(x_train,y_train,epochs = max_epochs,batch_size=ls,shuffle=True,validation_data=(val_x, val_y))   
    del x_train
    del y_train
    del history
    return model, lastLoss