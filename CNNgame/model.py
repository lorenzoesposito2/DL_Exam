import tensorflow as tf

def create_model(output_size=3):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=(50, 50, 4)),
        tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
        tf.keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(output_size , activation='linear')
    ])    
    # build method is used to create the model with the given input shape
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())
    return model
