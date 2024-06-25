import tensorflow as tf

def create_model(input_shape, hidden_size, output_size):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hidden_size, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(hidden_size, activation='relu'),
        tf.keras.layers.Dense(hidden_size, activation='relu'),
        tf.keras.layers.Dense(output_size)
    ])
    model.build(input_shape=(None, 11))
    return model




    