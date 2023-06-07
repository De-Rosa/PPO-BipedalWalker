using System.Collections.Generic;

namespace Physics.Walker.PPO;

// sequence of states and actions
public class Trajectory
{
    public List<Matrix> States;
    public List<Matrix> Actions;
    public List<Matrix> ActionProbabilities;
    public List<float> Rewards;
    public List<float> Returns;

    public Trajectory()
    {
        States = new List<Matrix>();
        Actions = new List<Matrix>();
        ActionProbabilities = new List<Matrix>();
        Rewards = new List<float>();
        Returns = new List<float>();
    }
}