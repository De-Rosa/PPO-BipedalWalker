namespace Physics.Walker.PPO;

// https://docs.google.com/document/d/1FZZvz0JMHKWOOVlXnrmeRMoGpyjqa0m6Q0S2qLECDpA
// https://spinningup.openai.com/en/latest/algorithms/ppo.html#pseudocode
// https://towardsdatascience.com/complete-guide-to-adam-optimization-1e5f29532c3d
// https://machinelearningmastery.com/adam-optimization-from-scratch/
// 2 layers, 64 neurons
// stochastic gradient descent using Adam optimizer

public class NeuralNetwork
{
    private Layer[] _layers;
    private DenseLayer[] _weights;
    
}