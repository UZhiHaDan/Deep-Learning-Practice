from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

# GRADED FUNCTION: model

def model(input_shape):
    """
    用 Keras 创建模型的图 Function creating the model's graph in Keras.
    
    参数：
    input_shape -- 模型输入数据的维度（使用Keras约定）
    
    返回：
    model -- Keras 模型实例
    """
    
    X_input = Input(shape = input_shape)
    
    # 第一步：卷积层 (≈4 lines)
    X = Conv1D(196, 15, strides=4)(X_input)             # CONV1D
    X = BatchNormalization()(X)                         # Batch normalization 批量标准化
    X = Activation('relu')(X)                           # ReLu activation ReLu 激活
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    # 第二步：第一个 GRU 层 (≈4 lines)
    X = GRU(units = 128, return_sequences=True)(X)      # GRU (使用128个单元并返回序列)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                         # Batch normalization 批量标准化

    # 第三步: 第二个 GRU 层  (≈4 lines)
    X = GRU(units = 128, return_sequences=True)(X)      # GRU (使用128个单元并返回序列)
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)
    X = BatchNormalization()(X)                         # Batch normalization 批量标准化
    X = Dropout(0.8)(X)                                 # dropout (use 0.8)

    # 第四步： 时间分布全连接层 (≈1 line)
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)

    model = Model(inputs = X_input, outputs = X)
    
    return model  

def detect_triggerword(filename):
    plt.subplot(2, 1, 1)

    x = graph_spectrogram(filename)
    # 频谱图输出（freqs，Tx），我们想要（Tx，freqs）输入到模型中
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions

chime_file = "audio_examples/chime.wav"
def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # 第一步：将连续输出步初始化为0
    consecutive_timesteps = 0
    # 第二步： 循环y中的输出步
    for i in range(Ty):
        # 第三步： 增加连续输出步
        consecutive_timesteps += 1
        # 第四步： 如果预测高于阈值并且已经过了超过75个连续输出步
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            # 第五步：使用pydub叠加音频和背景
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            # 第六步： 将连续输出步重置为0
            consecutive_timesteps = 0
        
    audio_clip.export("chime_output.wav", format='wav')
