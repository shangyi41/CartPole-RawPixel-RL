import torch as tc
from torch.nn import Module, Conv2d, Linear, Flatten, BatchNorm2d
from torch.nn.functional import relu

class DDQN(Module):
    """A Double Deep Q-Networks (DDQN) for CartPole."""

    def __init__(self, n_channels_in):
        """
        Create DDQN.
        
        Parameter
        --------------------
        n_channels_in: int
            number channels of input.
        """

        super(DDQN, self).__init__()

        # Input Layer = (84,84,4)
        # Conv. Layer = (20,20,32) ; kernel=8, stride=4, n_kernels=32
        # Conv. Layer = (9,9,64) ; kernel=4, stride=2, n_kernels=64
        # Conv. Layer = (7,7,64) ; kernel=3, stride=1, n_kernels=64
        # Fully Connected = 512 nodes
        # Output Layer = 2 nodes

        #Net's architecture.
        self.conv1 = Conv2d(n_channels_in, 32, kernel_size=8, stride=4)
        self.conv2 = Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = Flatten()
        self.fc = Linear(64*7*7, 512)
        self.out = Linear(512, 2)

        self.device = tc.device("cuda:0" if tc.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        """
        Evalue x.
        
        Parameter
        --------------------
        x: Tensor
            a tensor.

        Return
        --------------------
        y: Tensor
            x evalueted.
        """

        v = x / 255.0
        v = relu(self.conv1(v))
        v = relu(self.conv2(v))
        v = relu(self.conv3(v))
        v = self.flatten(v)
        v = relu(self.fc(v))
        y = self.out(v)

        return y

    def copy_from(self, other):
        """
        Copy parameters from another DDQN.
        
        Parameter
        --------------------
        other: DDQN
            a DDQN.
        """
        
        self.load_state_dict(other.state_dict())