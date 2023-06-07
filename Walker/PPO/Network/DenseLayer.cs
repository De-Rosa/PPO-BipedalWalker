using System.Numerics;

namespace Physics.Walker.PPO;

// https://analyticsindiamag.com/a-complete-understanding-of-dense-layers-in-neural-networks/
// weight matrix * values in previous layer
public class DenseLayer : Layer
{
    private Matrix _weights;
    private Matrix _biases;
}