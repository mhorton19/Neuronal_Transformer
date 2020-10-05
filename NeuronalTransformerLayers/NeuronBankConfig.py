class NeuronBankConfig():
    def __init__(self, num_neurons=100, values_len=10, num_heads=10, query_len=11, layer_norm_eps=1e-12, num_duplicates=5, num_iterations=3, use_connectivity=False):
        self.num_neurons = num_neurons
        self.values_len = values_len
        self.query_len = query_len
        self.num_heads = num_heads
        self.layer_norm_eps = layer_norm_eps
        self.num_duplicates = num_duplicates
        self.num_iterations = num_iterations
        self.use_connectivity = use_connectivity