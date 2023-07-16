using System.Collections.Generic;
using System.Linq;
using System.Transactions;

namespace Physics.Walker.PPO;

// sequence of states and actions
public class Trajectory
{
    public List<Matrix> States;
    public List<Matrix> Actions;
    public List<Matrix> LogProbabilities;

    public List<float> Advantages;
    public List<float> Rewards;
    public List<float> Returns;
    public List<float> Values;

    public List<int> Indexes;

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

    public Trajectory Copy()
    {
        Trajectory trajectory = new Trajectory
        {
            States = States.ToList(),
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