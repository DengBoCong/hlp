class Tacotron2Config(object):
    """Initialize Tacotron-2 Config."""

    def __init__(
            self,
            max_len=128,
            # TRAIN_SET_RATIO=0.2
            # vocab_size=5,
            embedding_hidden_size=512,
            initializer_range=0.02,
            layer_norm_eps=1e-6,
            embedding_dropout_prob=0.1,
            n_speakers=5,

            # encoder-conv1d层数
            n_conv_encoder=3,
            encoder_conv_filters=256,
            encoder_conv_kernel_sizes=5,
            encoder_conv_activation="relu",
            encoder_conv_dropout_rate=0.5,
            encoder_lstm_units=256,
            decoder_dim=256,
            decoder_lstm_dim=512,
            decoder_lstm_rate=0.1,

            # Attention parameters
            attention_dim=128,
            # Location Layer parameters
            attention_filters=32,
            attention_kernel=31,

            # Mel-post processing network parameters
            reduction_factor=5,
            n_prenet_layers=2,
            prenet_units=256,
            prenet_dropout_rate=0.5,
            n_lstm_decoder=1,
            decoder_lstm_units=512,
            gate_threshold=0.5,

            #postnet-conv1d层数
            n_conv_postnet=3,
            postnet_conv_filters=256,
            postnet_conv_kernel_sizes=5,
            postnet_dropout_rate=0.1,
            postnet_conv_activation="tanh",
            checkpoingt_dir=r"./checkpoints",

            # ljspeech的path
            wave_train_path=r"./data/LJSpeech-1.1/train/wavs/",
            wave_test_path=r"./data/LJSpeech-1.1/test/wavs/",
            csv_dir=r"./data/LJSpeech-1.1/metadata.csv",
            save_path_dictionary=r"./data/LJSpeech-1.1/dictionary.json",

            # number的path
            wave_train_path_number=r"./data/number/train/wavs/",
            wave_test_path_number=r"./data/number/test/wavs/",
            csv_dir_number=r"./data/number/metadata.csv",
            save_path_dictionary_number=r"./data/number/dictionary.json",

            # 关于音频的参数
            sr=22050,
            n_fft=2048,
            frame_shift=0.0125,
            frame_length=0.05,
            hop_length=275,
            win_length=1102,
            n_mels=80,
            power=1.2,
            n_iter=100,
            preemphasis=.97,
            max_db=100,
            ref_db=20,
            top_db=15,
            # 其他
            batch_size=32,
            #最大检查点保存数目
            max_to_keep=2,
    ):
        """tacotron2参数."""
        self.max_len = max_len
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

        # 解码器参数
        self.reduction_factor = reduction_factor
        self.n_prenet_layers = n_prenet_layers
        self.prenet_units = prenet_units
        self.prenet_dropout_rate = prenet_dropout_rate
        self.n_lstm_decoder = n_lstm_decoder
        self.decoder_lstm_units = decoder_lstm_units
        self.attention_dim = attention_dim
        self.attention_filters = attention_filters
        self.attention_kernel = attention_kernel
        self.n_mels = n_mels
        self.decoder_dim = decoder_dim
        self.decoder_lstm_dim = decoder_lstm_dim
        self.decoder_lstm_rate = decoder_lstm_rate
        self.gate_threshold = gate_threshold

        # postnet网络
        self.n_conv_postnet = n_conv_postnet
        self.postnet_conv_activation = postnet_conv_activation
        self.postnet_conv_filters = postnet_conv_filters
        self.postnet_conv_kernel_sizes = postnet_conv_kernel_sizes
        self.postnet_dropout_rate = postnet_dropout_rate

        # 检查点路径
        self.checkpoingt_dir = checkpoingt_dir

        # ljspeech路径
        self.wave_train_path = wave_train_path
        self.wave_test_path = wave_test_path
        self.save_path_dictionary = save_path_dictionary
        self.csv_dir = csv_dir

        # number路径
        self.wave_train_path_number = wave_train_path_number
        self.wave_test_path_number = wave_test_path_number
        self.save_path_dictionary_number = save_path_dictionary_number
        self.csv_dir_number = csv_dir_number

        # 声音参数
        self.sr = sr
        self.n_fft = n_fft
        self.frame_shift = frame_shift
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.power = power
        self.n_iter = n_iter
        self.preemphasis = preemphasis
        self.max_db = max_db
        self.ref_db = ref_db
        self.top_db = top_db

        # 其他
        self.batch_size = batch_size
        self.max_to_keep = max_to_keep
