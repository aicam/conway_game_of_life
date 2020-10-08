from keras.models import Sequential
from keras.optimizers import Adam
from ntm import NeuralTuringMachine as NTM


model = Sequential()


model.name = "NTM_-_"



ntm = NTM([625], n_slots=50, m_depth=20, shift_range=3,
          controller_model=None,
          return_sequences=True,
          input_shape=(None, 625),
          batch_size = 100)
model.add(ntm)

# sgd = Adam(lr=learning_rate, clipnorm=clipnorm)
model.compile(loss='binary_crossentropy', optimizer='Adam',
               metrics = ['binary_accuracy'], sample_weight_mode="temporal")
print(model.summary())