import pennylane as qml
from pennylane.operation import Operation, AnyWires

class ToffoliEntanglerLayers(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, weights, wires=None, rotation=None, do_queue=True, id=None):
        # convert weights to numpy array if weights is list otherwise keep unchanged
        interface = qml.math.get_interface(weights)
        weights = qml.math.asarray(weights, like=interface)

        shape = qml.math.shape(weights)
        if not (len(shape) == 3 or len(shape) == 2):  # 3 is when batching, 2 is no batching
            raise ValueError(
                f"Weights tensor must be 2-dimensional "
                f"or 3-dimensional if batching; got shape {shape}"
            )

        if shape[-1] != len(wires):
            # index with -1 since we may or may not have batching in first dimension
            raise ValueError(
                f"Weights tensor must have last dimension of length {len(wires)}; got {shape[-1]}"
            )

        self._hyperparameters = {"rotation": rotation or qml.RX}
        super().__init__(weights, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(weights, wires, rotation):  
        # first dimension of the weights tensor (second when batching) determines
        # the number of layers
        repeat = qml.math.shape(weights)[-2]

        op_list = []
        for layer in range(repeat):
            for i in range(len(wires)):
                op_list.append(rotation(weights[..., layer, i], wires=wires[i : i + 1]))
            
            if len(wires) < 3:
                raise ValueError(
                    f"ToffoliEntangler requires at least 3 wires.")
            
            elif len(wires) == 3:
                op_list.append(qml.Toffoli(wires=wires))

            elif len(wires) > 3:
                for i in range(len(wires)):
                    w = wires.subset([i, i + 1, i+2], periodic_boundary=True)
                    op_list.append(qml.Toffoli(wires=w))

        return op_list


    @staticmethod
    def shape(n_layers, n_wires):

        return n_layers, n_wires