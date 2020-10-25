class Tacotron2Config(object):
    """Initialize Tacotron-2 Config."""

    def __init__(
            self,
            embedding_hidden_size=256,
            prenet_units1=256,
            prenet_units2=128,
            K1=16,  # Size of the convolution bank in the encoder CBHG
            K2 = 8,  # Size of the convolution bank in the post processing CBHG
            filters=128,
            nb_layers=4,
            r=3,
            MAX_MEL_TIME_LENGTH=100,  # Maximum size of the time dimension for a mel spectrogram
            MAX_MAG_TIME_LENGTH=400 , # Maximum size of the time dimension for a spectrogram
            BATCH_SIZE=32,
            N_FFT=1024,
            nb_char_max=32,
            encoder_conv_dropout_rate=0.5,
            n_mels=80,
            PREEMPHASIS = 0.97,
            HOP_LENGTH = 800,
            WIN_LENGTH = 200,
            SAMPLING_RATE = 16000,
            N_MEL = 80,
            REF_DB = 20,
            MAX_DB = 100,
            TRAIN_SET_RATIO=0.9

    ):
        """Init parameters for Tacotron-2 model."""
        self.embedding_hidden_size = embedding_hidden_size
        self.prenet_units1 = prenet_units1
        self.prenet_units2 = prenet_units2
        self.k1 = K1
        self.k2 = K2
        self.filters = filters
        self.nb_layers = nb_layers
        self.r = r
        self.MAX_MEL_TIME_LENGTH = MAX_MEL_TIME_LENGTH  # Maximum size of the time dimension for a mel spectrogram
        self.MAX_MAG_TIME_LENGTH = MAX_MAG_TIME_LENGTH  # Maximum size of the time dimension for a spectrogram
        self.BATCH_SIZE = BATCH_SIZE
        self.N_FFT = N_FFT
        self.nb_char_max=nb_char_max
        self.encoder_conv_dropout_rate=encoder_conv_dropout_rate
        self.n_mels=n_mels
        self.PREEMPHASIS=PREEMPHASIS
        self.HOP_LENGTH=HOP_LENGTH
        self.WIN_LENGTH=WIN_LENGTH
        self.SAMPLING_RATE=SAMPLING_RATE
        self.N_MEL=N_MEL
        self.REF_DB=REF_DB
        self.MAX_DB=MAX_DB
        self.TRAIN_SET_RATIO=TRAIN_SET_RATIO