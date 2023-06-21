using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Physics.Bodies;
using Physics.Materials;
using Physics.Objects;
using Physics.Objects.RigidBodies;
using Physics.Rendering;
using Physics.Walker.PPO;
using Matrix = Physics.Walker.PPO.Matrix;

namespace Physics;

public class Environment
{
    private Walker.Walker _walker;
    private List<Walker.Walker> _simultaneousWalkers;
    
    private List<float> _rewards;
    private List<Trajectory> _trajectories;
    
    private bool _training = true;
    private int _steps = 0;
    private int _deadCount = 0;
    
    private readonly IMaterial _obstacleMaterial = new Metal();
    private const bool RoughFloor = false;

    private const string CriticFileLocation = "/Users/square/Projects/Physics/SavedModels/critic.txt";
    private const string ActorFileLocation = "/Users/square/Projects/Physics/SavedModels/actor.txt";
    private const int SimultaneousWalkerCount = 1;

    public Environment(List<IObject> rigidBodies)
    {
        _walker = new Walker.Walker();
        _simultaneousWalkers = new List<Walker.Walker>();
        
        CreateFloor(rigidBodies);
        ResetLists();
    }

    private void ResetLists()
    {
        _rewards = new List<float>();
        _trajectories = new List<Trajectory>();

        for (int i = 0; i < SimultaneousWalkerCount + 1; i++)
        {
            _rewards.Add(0);
            _trajectories.Add(new Trajectory());
        }
    }

    public void UpdateWalkers()
    {
        _steps += 1;
        
        for (int i = 0; i < _simultaneousWalkers.Count; i++)
        {
            _simultaneousWalkers[i].Update();
        }
    }

    public void TakeActions()
    {
        if (_steps == 1) return;
        
        for (int i = 0; i < _simultaneousWalkers.Count; i++)
        {
            TakeAction(_simultaneousWalkers[i], i);
        }
    }
    
    public void CheckStates(List<IObject> rigidBodies)
    {
        for (int i = 0; i < _simultaneousWalkers.Count; i++)
        {
            CheckState(rigidBodies, _simultaneousWalkers[i], i);
        }
    }

    public void StepWalkers(float deltaTime)
    {
        foreach (var joint in _simultaneousWalkers.SelectMany(walker => walker.GetJoints()))
        {
            joint.Step(deltaTime);
        }
    }

    public void RenderWalkers(Renderer renderer)
    {
        foreach (var walker in _simultaneousWalkers)
        {
            renderer.RenderJoint(walker.GetJointColors());
        }
    }

    public void Save()
    {
        _walker.Save(CriticFileLocation, ActorFileLocation);
    }

    public void Load()
    {
        _walker.Load(CriticFileLocation, ActorFileLocation);
    }

    private void TakeAction(Walker.Walker walker, int position)
    {
        Matrix state = walker.GetState();
        if (_training) _trajectories[position].States.Add(state);
        
        Matrix actions = walker.GetActions(state, out Matrix probabilities, out Matrix mean, out Matrix std);
        walker.TakeActions(actions);

        if (_training)
        {
            _trajectories[position].LogProbabilities.Add(probabilities);
            _trajectories[position].Means.Add(mean);
            _trajectories[position].Stds.Add(std);
            _trajectories[position].Actions.Add(actions);
        }
    }

    private void CheckState(List<IObject> rigidBodies, Walker.Walker walker, int position)
    {
        GetRewards();

        if (walker.GetPosition().X > 850)
        {
            _rewards[position] += 10;
            TerminalState(rigidBodies, walker, position);
            return;
        }

        if (walker.Terminal || _steps >= 10000)
        {
            _rewards[position] -= 10;
            TerminalState(rigidBodies, walker, position);
            return;
        }
        
        if (_training) _trajectories[position].Rewards.Add(_rewards[position]);
    }

    private void TerminalState(List<IObject> rigidBodies, Walker.Walker walker, int position)
    {
        _deadCount += 1;
        
        if (_training) _trajectories[position].Rewards.Add(_rewards[position]);
        
        Reset(walker, rigidBodies);
        walker.Terminal = false;
        
        if (_training && _deadCount == SimultaneousWalkerCount) StartTraining(rigidBodies);
    }

    private void GetRewards()
    {
        for (int i = 0; i < _simultaneousWalkers.Count; i++)
        {
            GetReward(_simultaneousWalkers[i], i + 1);
        }
    }
    
    private void GetReward(Walker.Walker walker, int position)
    {
        _rewards[position] = walker.GetChangeInPosition().X * 0.1f;
        if (walker.GetPosition().Y > 850)
        {
            _rewards[position] -= 0.0001f;
        }
    }
    
    private void StartTraining(List<IObject> rigidBodies)
    {
        TrainNetworks();
        ResetAll(rigidBodies);
    }

    private void TrainNetworks()
    {
        foreach (var trajectory in _trajectories)
        {
            _walker.Train(trajectory);
        }
    }

    private void Reset(Walker.Walker walker, List<IObject> rigidBodies)
    {
        walker.Reset(rigidBodies);
        _simultaneousWalkers.Remove(walker);
    }

    private void ResetAll(List<IObject> rigidBodies)
    {
        ResetLists();
        
        rigidBodies.Clear();
        _steps = 0;
        _deadCount = 0;
        
        CreateFloor(rigidBodies);
        CreateCreatures(rigidBodies);
    }

    public void CreateCreatures(List<IObject> rigidBodies)
    {
        for (int i = 0; i < SimultaneousWalkerCount; i++)
        {
            Walker.Walker walker = new Walker.Walker(_walker.GetBrain(), i);
            walker.CreateCreature(rigidBodies);
            _simultaneousWalkers.Add(walker);
        }
    }
    
    private void CreateFloor(List<IObject> rigidBodies, bool roughFloor = false)
    {
        if (roughFloor)
        {
            CreateRoughFloor(rigidBodies);
            return;
        }
        
        Vector2[] floorPositions = {
            new (-50, 1050), new (-50, 900), new (1050, 900), new (1050, 1050)
        };

        Hull floor = Hull.FromPositions(_obstacleMaterial, floorPositions, isStatic: true, isFloor: true);
        rigidBodies.Add(floor);
        
    }

    private void CreateRoughFloor(List<IObject> rigidBodies)
    {
        const int segments = 10;
        const int roughness = 100;
        
        int initialY = 800;
        int initialX = -50;
        
        Random random = new Random();
        Vector2 previousVector = new Vector2(initialX, initialY + random.Next(0, roughness));
        int movement = 1200 / segments;

        for (int i = 0; i < segments; i++)
        {
            int x = initialX + (i * movement);
            int y = 800 + random.Next(0, roughness);

            Vector2[] positions =
            {
                new(x, 1050), previousVector, new(x, y), new(x + movement, 1050)
            };
            
            Hull segment = Hull.FromPositions(_obstacleMaterial, positions, isStatic: true);
            rigidBodies.Add(segment);
                
            previousVector = new Vector2(x, y);
        }
    }
}