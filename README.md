# Neuronal_Transformer
This project attempts to use a transformer-like architecture to create a self-attention system inspired by biological neural networks

In standard transformers like BERT, self-attention is performed between token embeddings.  However, in the neuronal transformer, self-attention is performed between a 
set number of "neurons" each of which perform a different linear transformation to produce keys, queries and values.  This is inspired by biological neural networks which
use a set number of neurons to process information.  In this case, self-attention is used to connect neurons as opposed to synapses.

This approach seems to have a number of theoretical upsides.

1.  Neurons could potentially represent abstract concepts which can attend to tokens instead of strictly performing attention between tokens.
2.  The selective firing of shared neurons could potentially be an intuitive structure to perform multi-modal learning.
3.  Repeatedly performing self-attention across a single set of neurons could potentially allow for organic feedback connections, which are often seen in biological neural networks.
