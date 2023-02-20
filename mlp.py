import torch
from typing import List, Tuple
from torch import nn

class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.
       
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__()
        self.weights = nn.Parameter(torch.randn(in_features, out_features, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, out_features, requires_grad=True))
    def forward(self, input):
        """
            :param input: [bsz, in_features]
            :return result [bsz, out_features]
        """
        out = torch.matmul(input, self.weights) + self.bias
        return out


class MLP(torch.nn.Module):
    # 20 points
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int, activation: str = "relu"):
        super(MLP, self).__init__() 
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        assert len(hidden_sizes) > 1, "You should at least have one hidden layer"
        self.num_classes = num_classes
        self.activation = activation
        assert activation in ['tanh', 'relu', 'sigmoid'], "Invalid choice of activation"
        self.hidden_layers, self.output_layer = self._build_layers(input_size, hidden_sizes, num_classes)

        # Initializaton
        self._initialize_linear_layer(self.output_layer)
        for layer in self.hidden_layers:
            self._initialize_linear_layer(layer)
    
    def _build_layers(self, input_size: int, 
                        hidden_sizes: List[int], 
                        num_classes: int) -> Tuple[nn.ModuleList, nn.Module]:
        """
        Build the layers for MLP. Be ware of handlling corner cases.
        :param input_size: An int
        :param hidden_sizes: A list of ints. E.g., for [32, 32] means two hidden layers with 32 each.
        :param num_classes: An int
        :Return:
            hidden_layers: nn.ModuleList. Within the list, each item has type nn.Module
            output_layer: nn.Module
        """
        io_chain = [input_size] + hidden_sizes + [num_classes]
        layers = []
        for i in range(len(io_chain)-1): 
            print (io_chain[i], "-->", io_chain[i+1])
            layer = Linear(io_chain[i], io_chain[i+1])
            # layer = nn.Linear(io_chain[i], io_chain[i+1])
            layers.append(layer)
        hidden_layers = nn.ModuleList(layers[:-1])
        output_layer = layers[-1]
        return hidden_layers, output_layer
    
    def activation_fn(self, activation, inputs: torch.Tensor) -> torch.Tensor:
        """ process the inputs through different non-linearity function according to activation name """
        act = lambda x: x
        if activation == 'relu': act = nn.ReLU()
        elif activation == 'sigmoid': act = nn.Sigmoid()
        elif activation == 'softmax': act = nn.Softmax()
        else: return inputs
        return act(inputs)
        
        
    def _initialize_linear_layer(self, module: nn.Linear) -> None:
        """ For bias set to zeros. For weights set to glorot normal """
        nn.init.xavier_normal_(module.weights)
        module.bias.data.fill_(0)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """ Forward images and compute logits.
        1. The images are first fattened to vectors. 
        2. Forward the result to each layer in the self.hidden_layer with activation_fn
        3. Finally forward the result to the output_layer.
        
        :param images: [batch, channels, width, height]
        :return logits: [batch, num_classes]
        """
        layer_input = images.reshape(images.shape[0],-1)
        for hid_layer in self.hidden_layers:
            layer_input = hid_layer.forward(layer_input)
            layer_input = self.activation_fn(self.activation, layer_input)
        
        output = self.output_layer(layer_input)
        # output = self.activation_fn('softmax', output) if self.num_classes > 1 else self.activation_fn('sigmoid', output)
        # output = self.activation_fn(self.activation, output)
        return output
