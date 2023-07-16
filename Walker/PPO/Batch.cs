using System;

namespace Physics.Walker.PPO;

public class Batch
{
    public Matrix[] States;
    public Matrix[] LogProbabilities;
    public Matrix[] Actions;
    
    public float[] Rewards;
    public float[] Returns;
    public float[] Advantages;
    public float[] Values;

    public int[] Indexes;

    public Batch(int batchSize)
    {
        States = new Matrix[batchSize];
        LogProbabilities = new Matrix[batchSize];
        Actions = new Matrix[batchSize];
        Rewards = new float[batchSize];
        Returns = new float[batchSize];
        Values = new float[batchSize];
        Advantages = new float[batchSize];
        Indexes = new int[batchSize];
    }
}