import joblib
from tensorflow.keras.optimizers import Adam

from hparams import *
from tacotron_model import get_tacotron_model


def train(model_dir, data_dir='./data/'):
    decoder_input_training = joblib.load(data_dir + 'decoder_input_training.pkl')
    mel_spectro_training = joblib.load(data_dir + 'mel_spectro_training.pkl')
    spectro_training = joblib.load(data_dir + 'spectro_training.pkl')

    text_input_training = joblib.load(data_dir + 'text_input_ml_training.pkl')
    vocabulary = joblib.load(data_dir + 'vocabulary.pkl')

    model = get_tacotron_model(N_MEL, r, K1, K2, NB_CHARS_MAX,
                               EMBEDDING_SIZE, MAX_MEL_TIME_LENGTH,
                               MAX_MAG_TIME_LENGTH, N_FFT,
                               vocabulary)

    opt = Adam()
    model.compile(optimizer=opt,
                  loss=['mean_absolute_error', 'mean_absolute_error'])

    # TODO: 采用teacher-forcing方式进行训练
    model.fit([text_input_training, decoder_input_training],
                              [mel_spectro_training, spectro_training],
                              epochs=NB_EPOCHS, batch_size=BATCH_SIZE,
                              verbose=1, validation_split=0.15)

    model.save(model_dir + 'tts-model.h5')

train('./data/')