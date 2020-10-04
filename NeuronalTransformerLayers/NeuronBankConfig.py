class NeuronBankConfig():
    def __init__(self, num_neurons=100, vec_size=20, num_heads=5, layer_norm_eps=1e-12, num_duplicates=5, num_iterations=7, use_connectivity=False):
        self.num_neurons = num_neurons
        self.vec_size = vec_size
        self.num_heads = num_heads
        self.layer_norm_eps = layer_norm_eps
        self.num_duplicates = num_duplicates
        self.num_iterations = num_iterations
        self.use_connectivity = use_connectivity