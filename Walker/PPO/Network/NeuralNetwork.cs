using System;
using System.Collections.Generic;

namespace Physics.Walker.PPO.Network;

// Neural network class, the overall structure of multiple layers.
// Involves handling each layer during forward/back propagation and stores a cache for
// gradient calculation.
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
    
    // Adds a layer to the neural network.
    public void AddLayer(Layer layer)
    {
        _layers.Add(layer);
    }

    // Adds a dense layer to the neural network.
    public void AddLayer(DenseLayer layer)
    {
        _layers.Add(layer);
        _denseLayers.Add(layer);
    }

    // Adds multiple layers to the neural network.
    public void AddLayers(List<Layer> layers, List<DenseLayer> denseLayers)
    {
        foreach (var layer in layers)
        {   
            _layers.Add(layer);
        }

        foreach (var denseLayer in denseLayers)
        {
            _denseLayers.Add(denseLayer);
        }
    }

    // Feeds a matrix forward through the neural network.
    public Matrix FeedForward(Matrix matrix, bool cache = false)
    {
        if (cache) _cache.Clear();

        foreach (var layer in _layers)
        {
            if (cache) _cache.Add(matrix);
            matrix = layer.FeedForward(matrix);
        }

        return matrix;
    }

    // Feeds a matrix back through the neural network.
    public Matrix FeedBack(Matrix gradient)
    {
        for (int i = _layers.Count - 1; i >= 0; i--)
        {
            gradient = _layers[i].FeedBack(_cache[i], gradient);
        }

        return gradient;
    }

    // Optimises each dense layer in the neural network.
    public void Optimise()
    {
        foreach (var denseLayer in _denseLayers)
        {
            denseLayer.Adam();
        }
    }

    // Loads the neural network weights.
    public void Load(string type)
    {
        if (type != "critic" && type != "actor") return;

        string[] contents = type == "critic" ? Hyperparameters.CriticWeights : Hyperparameters.ActorWeights;
        if (contents.Length < 2 || contents == Array.Empty<string>()) return;

        string network = type == "critic" ? Hyperparameters.CriticNeuralNetwork : Hyperparameters.ActorNeuralNetwork;
        if (contents[0] != network) return;
        
        for (int i = 0; i < contents.Length - 1; i++)
        {
            _denseLayers[i].Load(contents[i + 1]);
        }
    }

    // Saves the neural network weights.
    public string[] Save(string type)
    {
        if (type != "critic" && type != "actor") return Array.Empty<string>();
        
        string[] contents = new string[_denseLayers.Count + 1];
        contents[0] = type == "critic" ? Hyperparameters.CriticNeuralNetwork : Hyperparameters.ActorNeuralNetwork;
        
        for (int i = 0; i < _denseLayers.Count; i++)
        {
            contents[i + 1] = _denseLayers[i].Save();
        }

        return contents;
    }
    
    // Zeros each dense layer.
    public void Zero()
    {
        foreach (var denseLayer in _denseLayers)
        {
            denseLayer.ZeroGradients();
        }
    }
}