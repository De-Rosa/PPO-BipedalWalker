using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Rendering;
using Vector2 = System.Numerics.Vector2;

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

    public void Render(Renderer renderer)
    {
        Vector2 position = new Vector2(50, 30);
        Vector2 layerGap = new Vector2(300, 0);
        Vector2 inputGap = new Vector2(0, 12);
        
        for (int i = 0; i < _denseLayers.Count; i++)
        {
            position += layerGap;
            Matrix weights = _denseLayers[i].GetWeights();
            Matrix cache = _cache[i];

            float weightMax = weights.AbsMax();
            float cacheMax = cache.AbsMax();

            for (int j = 0; j < weights.GetHeight(); j++)
            {
                for (int k = 0; k < weights.GetLength(); k++)
                {
                    Vector2 inputPosition = (position - layerGap) + (inputGap * k);
                    if (i == 0) inputPosition += inputGap * 24;
                    Vector2 outputPosition = position + (inputGap * j);
                    if (i == _denseLayers.Count - 1) outputPosition += inputGap * 30;

                    float weightsValue = weights.GetValue(j, k);
                    float weightsWhiteness = 255 * (Math.Abs(weightsValue) / (weightMax + 1e-10f));
                    
                    float cacheValue = cache.GetValue(k, 0);
                    float cacheWhiteness = 255 * (Math.Abs(cacheValue) / (cacheMax + 1e-10f));

                    int cacheWhitenessInt = Convert.ToInt32(cacheWhiteness);
                    int weightsWhitenessInt = Convert.ToInt32(weightsWhiteness);

                    if (cacheValue >= 0)
                    {
                        renderer.DrawLine(inputPosition, outputPosition, new Color(weightsWhitenessInt, weightsWhitenessInt + (cacheWhitenessInt / 4), weightsWhitenessInt), 1);
                    }
                    else
                    {
                        renderer.DrawLine(inputPosition, outputPosition, new Color(weightsWhitenessInt + (cacheWhitenessInt / 4), weightsWhitenessInt, weightsWhitenessInt), 1);
                    }
                    renderer.DrawSquare(inputPosition, 5, new Color(cacheWhitenessInt, cacheWhitenessInt, cacheWhitenessInt));
                }
            }
        }
    }

    public void Zero()
    {
        foreach (var denseLayer in _denseLayers)
        {
            denseLayer.ZeroGradients();
        }
    }

    private float Clip(float value, float upper, float lower)
    {
        if (value >= upper) return upper;
        return value <= lower ? lower : value;
    }
}