using System.Collections.Generic;
using System.Linq;

namespace Physics.Walker.PPO;

// Trajectory class, holds all information about one 'lifetime' for use in training.
public class Trajectory
{
    public List<Matrix> States { get; private init; }
    public List<Matrix> Actions { get; private init; }
    public List<Matrix> LogProbabilities { get; private init; }

    public List<float> Advantages { get; private init; }
    public List<float> Rewards { get; private init; }
    public List<float> Returns { get; private init; }
    public List<float> Values { get; private init; }

    public List<int> Indexes { get; private init; }

    public Trajectory()
    {
        States = new List<Matrix>();
        Actions = new List<Matrix>();
        LogProbabilities = new List<Matrix>();
        Rewards = new List<float>();
        Advantages = new List<float>();
        Returns = new List<float>();
        Values = new List<float>();
        Indexes = new List<int>();
    }

    // Creates a shallow copy of the current trajectory - meaning that all copied lists have references to the original.
    // This means that the matrices inside of copied trajectories cannot be edited since it would have side effects
    // for the original.
    public Trajectory Copy()
    {
        Trajectory trajectory = new Trajectory
        {
            States = States.GetRange(0, States.Count),
            Actions = Actions.ToList(),
            LogProbabilities = LogProbabilities.ToList(),
            Advantages = Advantages.ToList(),
            Rewards = Rewards.ToList(),
            Returns = Returns.ToList(),
            Values = Values.ToList(),
            Indexes = Indexes.ToList()
        };

        return trajectory;
    }
}