using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using NEA.Bodies;
using NEA.Materials;
using NEA.Objects;
using NEA.Objects.RigidBodies;
using NEA.Rendering;
using NEA.Walker.PPO;
using Matrix = NEA.Walker.PPO.Matrix;

namespace NEA;

// Environment class holds the walker and it's environment.
// Responsible for the update loop and the tracking of data of the walker.
public class Environment
{
    // Variables from Game class, for use in rendering
    private readonly SpriteBatch _spriteBatch;
    private readonly Renderer _renderer;
    
    private readonly Walker.Walker _walker;
    private Trajectory _trajectory;
    
    private readonly List<RigidBody> _rigidBodies;
    
    // Training information used for console display
    private int _steps = 0;
    private int _episodes = 0;
    private float _bestDistance = 0;
    private float _previousAverageReward = 0;
    
    private Walker.PPO.Matrix _state;
    private (bool, bool) _legsOnFloor;

    // Environment class houses the physics/AI loop and stores the trajectory/training information.
    public Environment(SpriteBatch spriteBatch, Renderer renderer)
    {
        _spriteBatch = spriteBatch;
        _renderer = renderer;
        _rigidBodies = new List<RigidBody>();
        _legsOnFloor = (true, true);
        
        _walker = new Walker.Walker();
        _walker.CreateCreature(_rigidBodies);
        CreateFloor();

        _trajectory = new Trajectory();
    }
    
    // Returns information pertinent to the console during training.
    // Output of the function is formatted as follows:
    // (int episode, int timeStep, float distance, float averageReward, float bestDistance, float pastAverageReward)
    public (int, int, float, float, float, float, Walker.PPO.Matrix) GetConsoleInformation()
    {
        float averageReward = _trajectory.Rewards.Count == 0 ? 0 : _trajectory.Rewards.Average();
        return (_episodes, _steps, _walker.GetPosition().X, averageReward, _bestDistance, _previousAverageReward, _state);
    }
    
    // Environment update function, follows training loop:
    // Observe state -> Take Action -> Step Physics -> Receive Reward
    public void Update(float deltaTime)
    {
        if (Console.KeyAvailable && Console.ReadKey(true).Key == ConsoleKey.X)
        {
           _renderer.ExitTraining();
        }
        
        _trajectory.Indexes.Add(_steps);
        _steps++;
        
        _trajectory.States.Add(_state);
        Walker.PPO.Matrix action = _walker.GetActions(_state, out Walker.PPO.Matrix logProbabilities);
        try
        {
            _walker.TakeActions(Matrix.Clip(action, 1f, -1f));
        }
        catch (Exception e)
        {
            ErrorLogger.LogError($"Exception occurred while attempting to clip the action matrix during the environment update: {e.Message}");
        }

        _state = Step(deltaTime, out float reward, out bool terminal);
        
        _trajectory.Actions.Add(action);
        _trajectory.LogProbabilities.Add(logProbabilities);
        _trajectory.Rewards.Add(reward);

        if (terminal) TrainNetworks();
    }

    // Steps physics and returns a reward using a reward function.
    // Terminal returns true when the walker's head collides with the ground or the timestep count exceeds the maximum.
    private Walker.PPO.Matrix Step(float deltaTime, out float reward, out bool terminal)
    {
        terminal = false;

        StepObjects(deltaTime);
        _walker.Update();
        
        reward = CalculateReward();
        
        // Fail
        if (_walker.Terminal || _steps > Hyperparameters.MaxTimesteps)
        {
            if (_walker.Terminal) reward -= 40f;
            terminal = true;
        }

        // Success
        if (_walker.GetPosition().X > 900)
        {
            reward += 80f;
            terminal = true;
        }

        if (_walker.GetPosition().X > _bestDistance) _bestDistance = _walker.GetPosition().X;

        return _walker.GetState();
    }

    // Physics step, updates every rigid body and joint in the environment.
    // Is subdivided according to the Iteration setting, providing better stability.
    public void StepObjects(float deltaTime)
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
    
    // Reward function, provides a positive reward when the walker moves right (above a certain level) and a negative reward when too low.
    // We don't discourage moving backwards as the AI may enter a local minimum where it attempts to kill itself as quickly as possible
    // to not receive negative rewards.
    private float CalculateReward()
    {
        float reward = 0;
        reward += (_walker.GetChangeInPosition().X > 0 && (((_walker.GetJoints()[0].GetPointA().Y) / 500f) < 1.6f)) ? _walker.GetChangeInPosition().X : 0; // horizontal movement
        reward -= ((_walker.GetJoints()[0].GetPointA().Y) / 500f) > 1.65f ? -0.1f : 0; // body too low, negative reward
        return reward;
    }

    // Trains the walker's AI, and stores information to use in the console.
    private void TrainNetworks()
    {
        _episodes++;
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
        
        // Render the walker's bodies
        RenderWalker();
        
        // Draw the current best distance.
        _renderer.DrawLine(new Vector2(_bestDistance, 900) + _renderer.GetCameraPosition(),
            new Vector2(_bestDistance, 500) + _renderer.GetCameraPosition(),
            Color.Lime, 2f);
        
        // Render each rigid body in the environment
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
    private void CreateRoughFloor(int segments = 10, int roughness = 100)
    {
        if (segments <= 0 || roughness < 0)
        {
            ErrorLogger.LogError("Invalid floor segments/roughness values.");
            CreateRoughFloor();
            return;
        }
        
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