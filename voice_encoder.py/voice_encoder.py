
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv2D, BatchNormalization, Dropout, Bidirectional, LSTM, TimeDistributed, Dense, GlobalAveragePooling1D
import librosa
import numpy as np

# 音響特徴の抽出
def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=16000)
    # print(signal, sr)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(signal)
    spectral_flux = librosa.onset.onset_strength(y=signal, sr=sr)
    # print(mfcc,zcr,spectral_flux)
    f0, voiced_flag, voiced_probs = librosa.pyin(signal, sr=sr, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

    # print(f0,voiced_flag,voiced_probs)
    return mfcc, zcr, spectral_flux, f0

# モデルの構築
def build_model(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # 畳み込みブロック
    x = Conv1D(256, 3, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # 注意機構の追加
    attention_data = tf.keras.layers.Attention()([x, x])

    # BLSTM層
    x = Bidirectional(LSTM(64, return_sequences=True))(attention_data)

    # 時間分布層
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.4)(x)

    # 出力層
    outputs = Dense(5, activation='sigmoid')(x)  # 5つの吃音イベントを分類

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# 特徴の取得
mfcc, zcr, spectral_flux, f0 = extract_features('stutter-i-love-you-82913.wav')
spectral_flux_reshaped = np.expand_dims(spectral_flux, axis=0)  

f0_reshaped = np.expand_dims(f0, axis=0) 
print(f0_reshaped)
# 特徴の結合と前処理
features = np.concatenate([mfcc, zcr, spectral_flux_reshaped, f0_reshaped], axis=0)
features = np.expand_dims(features, axis=0)  # バッチ次元を追加

# モデルのインスタンス化と訓練
model = build_model(features.shape[1:])
model.summary()

# ダミーデータによる訓練（実際には適切なデータセットを使用）
# X_train, y_trainは準備されたデータセットを使用
# model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

