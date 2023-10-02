using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Physics.Bodies;
using Physics.Materials;
using Physics.Objects;
using Physics.Objects.RigidBodies;
using Physics.Rendering;
using Physics.Walker.PPO;
using Matrix = Physics.Walker.PPO.Matrix;

namespace Physics;

// Environment class holds the walker and it's environment.
// Responsible for the update loop and the tracking of data of the walker.
public class Environment
{
    // Variables from Game class, for use in rendering
    private readonly SpriteBatch _spriteBatch;
    private readonly Renderer _renderer;
    
    private readonly Walker.Walker _walker;
    private Trajectory _trajectory;
    
    private List<RigidBody> _rigidBodies;
    
    // Training information used for console display
    private int _steps = 0;
    private int _episodes = 0;
    private float _bestDistance = 0;
    private float _previousAverageReward = 0;
    
    private Matrix _state;
    
    // Environment class houses the physics/AI loop and stores the trajectory/training information.
    public Environment(SpriteBatch spriteBatch, Renderer renderer)
    {
        _spriteBatch = spriteBatch;
        _renderer = renderer;
        _rigidBodies = new List<RigidBody>();
        
        _walker = new Walker.Walker();
        _walker.CreateCreature(_rigidBodies);
        CreateFloor();

        _trajectory = new Trajectory();
    }
    
    // Returns information pertinent to the console during training.
    // Output of the function is formatted as follows:
    // (int episode, int timeStep, float distance, float averageReward, float bestDistance, float pastAverageReward)
    public (int, int, float, float, float, float, Matrix) GetConsoleInformation()
    {
        float averageReward = _trajectory.Rewards.Count == 0 ? 0 : _trajectory.Rewards.Average();
        return (_episodes, _steps, _walker.GetPosition().X, averageReward, _bestDistance, _previousAverageReward, _state);
    }
    
    // Environment update function, follows training loop:
    // Observe state -> Take Action -> Step Physics -> Receive Reward
    public void Update(float deltaTime)
    {
        _trajectory.Indexes.Add(_steps);
        _steps++;
        
        _trajectory.States.Add(_state);
        Matrix action = _walker.GetActions(_state, out Matrix logProbabilities);
        _walker.TakeActions(Matrix.Clip(action, 1f, -1f));

        _state = Step(deltaTime, out float reward, out bool terminal);
        
        _trajectory.Actions.Add(action);
        _trajectory.LogProbabilities.Add(logProbabilities);
        _trajectory.Rewards.Add(reward);

        if (terminal) TrainNetworks();
    }

    // Steps physics and returns a reward using a reward function.
    // Terminal returns true when the walker's head collides with the ground or the timestep count exceeds the maximum.
    private Matrix Step(float deltaTime, out float reward, out bool terminal)
    {
        terminal = false;

        StepObjects(deltaTime);
        _walker.Update();
        
        reward = CalculateReward();
        if (_walker.Terminal || _steps > Hyperparameters.MaxTimesteps)
        {
            reward -= 100f;
            terminal = true;
        }
        
        return _walker.GetState();
    }

    // Physics step, updates every rigid body and joint in the environment.
    // Is subdivided according to the Iteration setting, providing better stability.
    private void StepObjects(float deltaTime)
    {
        deltaTime /= Hyperparameters.Iterations;
        
        for (int i = 0; i < Hyperparameters.Iterations; i++)
        {
            foreach (var joint in _walker.GetJoints())
            {
                joint.Step();
            }
            
            foreach (var body in _rigidBodies)
            {
                var rigidObject = (IObject) body;
                rigidObject.Update(_rigidBodies, deltaTime);
            };
        }
    }
    
    // Reward function, provides a positive reward when the walker moves right.
    // By including negative rewards, the AI may enter a local minimum where it attempts to kill itself as quickly as possible
    // to not receive negative rewards.
    private float CalculateReward()
    {
        if (_walker.GetChangeInPosition().X < 0) return 0;
        return _walker.GetChangeInPosition().X;
    }

    // Trains the walker's AI, and stores information to use in the console.
    private void TrainNetworks()
    {
        _episodes++;
        if (_walker.GetPosition().X > _bestDistance) _bestDistance = _walker.GetPosition().X;
        _previousAverageReward = _trajectory.Rewards.Average();
        _walker.Train(_trajectory, _renderer);

        Reset();
    }
    
    // Resets variables relating to each trajectory.
    private void Reset()
    {
        _trajectory = new Trajectory();
        _steps = 0;
        _walker.Reset(_rigidBodies);
        InitialState();
    }

    // Sets values for the initial state of the walker.
    public void InitialState()
    {
        _walker.Update();
        _state = _walker.GetState();
    }
    
    // Draws the joints of the walker and their color, corresponding to their current torque.
    public void RenderWalker()
    {
        _renderer.RenderJoints(_walker.GetJointColors());
    }
    
    // Draw each object using the Renderer class.
    public void RenderObjects()
    {
        _spriteBatch.Begin();
        
        RenderWalker();
        
        foreach (var iObject in _rigidBodies)
        {
            _renderer.RenderRigidObject(iObject);
        }

        _spriteBatch.End();
    }
    
    // Creates a static floor rigid body.
    private void CreateFloor()
    {
        if (Hyperparameters.RoughFloor)
        {
            CreateRoughFloor();
            return;
        }
        
        Vector2[] floorPositions = {
            new (-50, 1050), new (-50, 900), new (1050, 900), new (1050, 1050)
        };

        Hull floor = Hull.FromPositions(new Metal(), floorPositions, isStatic: true, isFloor: true);
        _rigidBodies.Add(floor);
        
    }

    // Creates multiple static rigid bodies which varies in height to create an uneven floor.
    // The floor must be subdivided, since the physics engine doesn't support concave shapes.
    private void CreateRoughFloor()
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
            
            Hull segment = Hull.FromPositions(new Metal(), positions, isStatic: true, isFloor: true);
            _rigidBodies.Add(segment);
                
            previousVector = new Vector2(x, y);
        }
    }
}