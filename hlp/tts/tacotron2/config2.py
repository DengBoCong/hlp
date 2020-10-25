class Tacotron2Config(object):
    """Initialize Tacotron-2 Config."""
    def __init__(
        self,
        vocab_size=5,
        embedding_hidden_size=512,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        embedding_dropout_prob=0.1,
        n_speakers=5,
        n_conv_encoder=3,
        encoder_conv_filters=512,
        encoder_conv_kernel_sizes=5,
        encoder_conv_activation="mish",
        encoder_conv_dropout_rate=0.5,
        encoder_lstm_units=256,
        decoder_dim=256,
        decoder_lstm_dim=1024,
        decoder_lstm_rate=0.1,

        # Attention parameters
        attention_dim=128,

        # Location Layer parameters
        attention_filters=32,
        attention_kernel=31,

        # Mel-post processing network parameters
        # postnet_embedding_dim=512,
        # postnet_kernel_size=5,
        # postnet_n_convolutions=5,


        reduction_factor=5,
        n_prenet_layers=2,
        prenet_units=256,
        prenet_activation="mish",
        prenet_dropout_rate=0.5,
        n_lstm_decoder=1,
        decoder_lstm_units=1024,
        attention_type="lsa",

        gate_threshold=0.5,
        n_conv_postnet=5,
        postnet_conv_filters=512,
        postnet_conv_kernel_sizes=5,
        postnet_dropout_rate=0.1,
        #path
        text_train_path=r"./number/train/text/wenzi.txt",
        wave_train_path= r"./number/train/wave/",
        #text_test_path= r".\text\text_test\wenzi.txt",
        #wave_test_path = r"./wavs/wave_test/",
        #关于音频的参数
        sr=22050,# Sample rate.
        n_fft = 2048,  # fft points (samples)
        frame_shift = 0.0125,  # seconds
        frame_length = 0.05,  # seconds
        #hop_length = int(sr*frame_shift),  # samples.
        hop_length=275,
        #win_length = int(sr*frame_length),  # samples.
        win_length = 1102,
        n_mels = 80,  # Number of Mel banks to generate
        power = 1.2,  # Exponent for amplifying the predicted magnitude
        n_iter = 100,  # Number of inversion iterations
        preemphasis = .97,  # or None
        max_db = 100,
        ref_db = 20,
        top_db = 15,
        batch_size=2

    ):
        """Init parameters for Tacotron-2 model."""
        self.embedding_hidden_size = embedding_hidden_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.embedding_dropout_prob = embedding_dropout_prob
        self.n_speakers = n_speakers
        self.n_conv_encoder = n_conv_encoder
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_sizes = encoder_conv_kernel_sizes
        self.encoder_conv_activation = encoder_conv_activation
        self.encoder_conv_dropout_rate = encoder_conv_dropout_rate
        self.encoder_lstm_units = encoder_lstm_units
        self.vocab_size=vocab_size

        # decoder param
        self.reduction_factor = reduction_factor
        self.n_prenet_layers = n_prenet_layers
        self.prenet_units = prenet_units
        self.prenet_activation = prenet_activation
        self.prenet_dropout_rate = prenet_dropout_rate
        self.n_lstm_decoder = n_lstm_decoder
        self.decoder_lstm_units = decoder_lstm_units
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.attention_filters = attention_filters
        self.attention_kernel = attention_kernel
        self.n_mels = n_mels
        self.decoder_dim=decoder_dim
        self.decoder_lstm_dim=decoder_lstm_dim
        self.decoder_lstm_rate=decoder_lstm_rate
        self.gate_threshold = gate_threshold
        # postnet
        self.n_conv_postnet = n_conv_postnet
        self.postnet_conv_filters = postnet_conv_filters
        self.postnet_conv_kernel_sizes = postnet_conv_kernel_sizes
        self.postnet_dropout_rate = postnet_dropout_rate
        # path
        self.text_train_path=text_train_path
        self.wave_train_path = wave_train_path
#        self.text_test_path = text_test_path
 #       self.wave_test_path = wave_test_path
        self.batch_size=batch_size
        # 声音参数
        self.sr=sr
        self.n_fft=n_fft
        self.frame_shift=frame_shift
        self.frame_length=frame_length
        self.hop_length=hop_length
        self.win_length=win_length
        self.n_mels=n_mels
        self.power=power
        self.n_iter=n_iter
        self.preemphasis=preemphasis
        self.max_db=max_db
        self.ref_db=ref_db
        self.top_db=top_db

