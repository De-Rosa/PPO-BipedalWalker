using System;
using System.Collections.Generic;
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
    private Trajectory _trajectory;
    
    private int _steps = 0;
    
    private readonly IMaterial _obstacleMaterial = new Metal();
    private const bool RoughFloor = false;

    private const string CriticFileLocation = "/Users/square/Projects/Physics/SavedModels/critic.txt";
    private const string ActorFileLocation = "/Users/square/Projects/Physics/SavedModels/actor.txt";
    private const string DataFileLocation = "/Users/square/Projects/Physics/Data/data.txt";
    private const int DataLength = 1;

    public Environment(List<RigidBody> rigidBodies)
    {
        _walker = new Walker.Walker();
        _walker.CreateCreature(rigidBodies);
        CreateFloor(rigidBodies);

        _trajectory = new Trajectory();
    }
    
    public void Update(List<RigidBody> rigidBodies)
    {
        _walker.Update();
        _steps++;
        float reward = CalculateReward();

        Matrix nextState = _walker.GetState();

        if (_walker.Terminal || _steps > 10000)
        {
            _trajectory.Rewards.Add(reward);
            TrainNetworks();
            Reset(rigidBodies);
            return;
        }
        
        Matrix nextAction = _walker.GetActions(nextState, out Matrix logProbabilities);
        
        _walker.TakeActions(Matrix.Clip(nextAction, 1f, -1f));
        
        _trajectory.Rewards.Add(reward);
        _trajectory.States.Add(nextState);
        _trajectory.Actions.Add(nextAction);
        _trajectory.LogProbabilities.Add(logProbabilities);
        _trajectory.Indexes.Add(_trajectory.Indexes.Count);
    }

    private float CalculateReward()
    {
        float reward = 0;
        float distance = _walker.GetChangeInPosition().X;
        
        if (distance > 0) reward += distance;
        if (_walker.Terminal || _steps > 10000) reward -= 100;

        return reward;
    }
    
    public void RenderWalker(Renderer renderer)
    {
        renderer.RenderJoint(_walker.GetJointColors());
    }

    public void StepWalker(float deltaTime)
    {
        foreach (var joint in _walker.GetJoints())
        {
            joint.Step(deltaTime);
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
    
    private void TrainNetworks()
    {
        _trajectory.Rewards.RemoveAt(0);
        _walker.Train(_trajectory);
    }

    private void Reset(List<RigidBody> rigidBodies)
    {
        _trajectory = new Trajectory();
        _steps = 0;
        _walker.Reset(rigidBodies);
    }

    private void CreateFloor(List<RigidBody> rigidBodies, bool roughFloor = false)
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

    private void CreateRoughFloor(List<RigidBody> rigidBodies)
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