import numpy as np

'''deltas = self.get_deltas(old_weights, weights)

        deltas_dictionary = self.get_flat(deltas)

        deltas_dictionary['vector'], deltas_dictionary['scale'] = self.uniform_quantization(deltas_dictionary['vector'])

        encoded_deltas = self.rle_encoding(deltas_dictionary['vector'])

        deltas_dictionary['vector'] = encoded_deltas'''


def decode_rle(encoded):

    #print("DECODE RLE WAS CALLED")

    sentinel = -128
    decoded = []
    if isinstance(encoded, list):
        values = encoded
    else:
        values = encoded.tolist()

    i = 0

    while i < len(values):
        if values[i] == sentinel:
            sequence_length = int(values[i + 1])
            decoded.extend([0] * sequence_length)
            i += 2
        else:
            decoded.append(values[i])
            i += 1

    return np.array(decoded, dtype=np.int8)

def dequantize(quantized_vector: np.ndarray, scale: float) -> np.ndarray:

    """Uses the Scale value that was sent along the the encoded vector to recover weights differences (deltas) as close as possible"""
    
    return quantized_vector.astype(np.float32) * scale

def unflatten(flat_vector: np.ndarray, metadata: dict) -> list:

    """After dequantization the vector is turned back to a list for keras to be able to set the weights of the model"""

    #avoids errors if metadata end up as strings at some point
    metadata = {int(k): v for k, v in metadata.items()} 

    layers = []
    offset = 0
 
    for index in sorted(metadata.keys()):
        shape = metadata[index]['shape']
        size  = metadata[index]['size']
 
        layer_flat = flat_vector[offset : offset + size]
        layers.append(layer_flat.reshape(shape))
 
        offset += size
 
    return layers

def apply_deltas(global_weights: list, deltas: list) -> list:

    """Adds the deltas to the old global weights to recover the updated weights"""

    return [g.astype(np.float32) + d.astype(np.float32) for g, d in zip(global_weights, deltas)]

def delta_decompress(deltas_dictionary, global_weights):
    """
    This function is responsible for taking the deltas_dictionary from the client updates, decoding and dequantizing it, and applying the deltas to the global weights to recover the updated weights.
    """

    decoded = decode_rle(deltas_dictionary['vector'])
    scale = deltas_dictionary['scale']
    dequantized = dequantize(decoded, scale)
    flat = unflatten(dequantized, deltas_dictionary['metadata'])
    rebuilt_weights = apply_deltas(global_weights, flat)

    return rebuilt_weights