import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ข้อมูลขนาดของลูกตา
eye_sizes = [
    1500, 1600, 1650, 1700, 1750,
    1800, 1850, 1900, 1950, 2000,
    2050, 2100, 2150, 2200, 2250,
    2300, 2350, 2400, 2450, 2500
]

# ฟังก์ชันสำหรับสร้างชุดข้อมูลสำหรับการฝึกโมเดล
def generate_data(eye_sizes, sequence_length):
    X = []
    y = []
    for i in range(len(eye_sizes) - sequence_length):
        X.append(eye_sizes[i:i+sequence_length])
        y.append(eye_sizes[i+sequence_length])
    return np.array(X), np.array(y)

# สร้างชุดข้อมูล
sequence_length = 5  # ปรับตามจำนวนข้อมูลที่คุณต้องการใช้ในการทำนาย
X, y = generate_data(eye_sizes, sequence_length)

# ปรับข้อมูลให้อยู่ในรูปแบบที่เหมาะสมสำหรับโมเดล
X = (X - np.mean(X)) / np.std(X)
y = (y - np.mean(y)) / np.std(y)

# แบ่งชุดข้อมูลเป็นชุดฝึกและชุดทดสอบ
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# กำหนดโมเดล
model = models.Sequential([
    layers.LSTM(64, input_shape=(sequence_length, 1)),
    layers.Dense(1)
])

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='mse')

# ปรับรูปแบบข้อมูลสำหรับการนำเข้า LSTM (batch_size, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], sequence_length, 1))
X_test = np.reshape(X_test, (X_test.shape[0], sequence_length, 1))

# เทรนโมเดล
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# ประเมินโมเดล
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# ทำนาย
predictions = model.predict(X_test)

# หาค่าต่ำสุดและค่าสูงสุดของการทำนาย
min_prediction = np.min(predictions)
max_prediction = np.max(predictions)

# ปรับค่าทำนายให้เป็นหน่วยปกติ
mean_eye_size = np.mean(eye_sizes)
std_eye_size = np.std(eye_sizes)
normalized_min_prediction = min_prediction * std_eye_size + mean_eye_size
normalized_max_prediction = max_prediction * std_eye_size + mean_eye_size

print(f"Predicted range of eye sizes: from {normalized_min_prediction} to {normalized_max_prediction}")
