using System;
using System.Collections.Generic;

namespace Physics.Walker.PPO;

// https://docs.google.com/document/d/1FZZvz0JMHKWOOVlXnrmeRMoGpyjqa0m6Q0S2qLECDpA
// https://spinningup.openai.com/en/latest/algorithms/ppo.html#pseudocode
// https://towardsdatascience.com/complete-guide-to-adam-optimization-1e5f29532c3d
// https://machinelearningmastery.com/adam-optimization-from-scratch/
// stochastic gradient descent using Adam optimizer

public class NeuralNetwork
{
    private readonly List<Layer> _layers;
    private readonly List<DenseLayer> _denseLayers;
    private readonly List<Matrix> _cache;

    public NeuralNetwork()
    {
        _layers = new List<Layer>();
        _denseLayers = new List<DenseLayer>();
        _cache = new List<Matrix>();
    }
    
    public void AddLayer(Layer layer)
    {
        _layers.Add(layer);
    }

    public void AddLayer(DenseLayer layer)
    {
        _layers.Add(layer);
        _denseLayers.Add(layer);
    }

    public Matrix FeedForward(Matrix matrix)
    {
        _cache.Clear();
        
        foreach (var layer in _layers)
        {
            _cache.Add(matrix);
            matrix = layer.FeedForward(matrix);
        }

        return matrix;
    }

    public Matrix FeedBack(Matrix gradient)
    {
        for (int i = _layers.Count - 1; i >= 0; i--)
        {
            gradient = _layers[i].FeedBack(_cache[i], gradient);
        }

        return gradient;
    }

    public void Optimise()
    {
        foreach (var denseLayer in _denseLayers)
        {
            denseLayer.Adam();
        }

        foreach (var denseLayer in _denseLayers)
        {
            denseLayer.ZeroGradients();
        }
    }

    public NeuralNetwork Clone()
    {
        NeuralNetwork newNetwork = new NeuralNetwork();
        for (int i = 0; i < _layers.Count; i++)
        {
            Layer clonedLayer = _layers[i].Clone();
            if (_layers[i].GetType() == LayerType.DENSE)
            {
                newNetwork.AddLayer((DenseLayer) clonedLayer);
            }
            else
            {
                newNetwork.AddLayer(clonedLayer);
            }
        }

        return newNetwork;
    }

    public void Load(string[] contents)
    {
        for (int i = 0; i < contents.Length; i++)
        {
            _denseLayers[i].Load(contents[i]);
        }
    }

    public string[] Save()
    {
        string[] contents = new string[_denseLayers.Count];
        for (int i = 0; i < _denseLayers.Count; i++)
        {
            contents[i] = _denseLayers[i].Save();
        }

        return contents;
    }

    public void Zero()
    {
        foreach (var denseLayer in _denseLayers)
        {
            denseLayer.ZeroGradients();
        }
    }
}