using System.Collections.Generic;
using System.Linq;
using System.Transactions;

namespace Physics.Walker.PPO;

// sequence of states and actions
public class Trajectory
{
    public List<Matrix> States;
    public List<Matrix> Actions;
    public List<Matrix> Probabilities;
    public List<float> Advantages;
    public List<float> Rewards;
    public List<float> Returns;
    public List<float> Values;

    public Trajectory()
    {
        States = new List<Matrix>();
        Actions = new List<Matrix>();
        Probabilities = new List<Matrix>();
        Rewards = new List<float>();
        Advantages = new List<float>();
        Returns = new List<float>();
        Values = new List<float>();
    }

    public Trajectory Copy()
    {
        Trajectory trajectory = new Trajectory
        {
            States = States.ToList(),
            Actions = Actions.ToList(),
            Probabilities = Probabilities.ToList(),
            Advantages = Advantages.ToList(),
            Rewards = Rewards.ToList(),
            Returns = Returns.ToList(),
            Values = Values.ToList()
        };

        return trajectory;
    }
}