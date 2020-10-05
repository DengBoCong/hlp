class Tacotron2Config(object):
    """Initialize Tacotron-2 Config."""
    def __init__(
        self,
        embedding_hidden_size=512,
        embedding_dropout_prob=0.1,
        encoder_conv_filters=512,
        encoder_conv_kernel_sizes=5,
        encoder_conv_dropout_rate=0.5,
        encoder_lstm_units=256,
        decoder_dim=256,
        decoder_lstm_dim=1024,
        decoder_lstm_rate=0.1,

        reduction_factor=1,
        n_prenet_layers=2,
        prenet_units=256,
        prenet_dropout_rate=0.5,
        decoder_lstm_units=1024,
        attention_type="lsa",
        attention_dim=128,
        attention_filters=64,
        attention_kernel=5,
        n_mels=80,
        n_conv_postnet=5,
        postnet_conv_filters=512,
        postnet_conv_kernel_sizes=5,
        postnet_dropout_rate=0.1,
    ):
        """Init parameters for Tacotron-2 model."""
        self.embedding_hidden_size = embedding_hidden_size
        self.embedding_dropout_prob = embedding_dropout_prob
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_sizes = encoder_conv_kernel_sizes
        self.encoder_conv_dropout_rate = encoder_conv_dropout_rate
        self.encoder_lstm_units = encoder_lstm_units

        # decoder param
        self.reduction_factor = reduction_factor
        self.n_prenet_layers = n_prenet_layers
        self.prenet_units = prenet_units
        self.prenet_dropout_rate = prenet_dropout_rate
        self.decoder_lstm_units = decoder_lstm_units
        self.attention_type = attention_type
        self.attention_dim = attention_dim
        self.attention_filters = attention_filters
        self.attention_kernel = attention_kernel
        self.n_mels = n_mels
        self.decoder_dim=decoder_dim
        self.decoder_lstm_dim=decoder_lstm_dim
        self.decoder_lstm_rate=decoder_lstm_rate

        # postnet
        self.n_conv_postnet = n_conv_postnet
        self.postnet_conv_filters = postnet_conv_filters
        self.postnet_conv_kernel_sizes = postnet_conv_kernel_sizes
        self.postnet_dropout_rate = postnet_dropout_rate
