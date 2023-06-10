using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Physics.Bodies;
using Physics.Bodies.Physics;
using Physics.Input;
using Physics.Materials;
using Physics.Objects;
using Physics.Objects.RigidBodies;
using Physics.Objects.SoftBodies;
using Physics.Rendering;
using Physics.Walker.PPO;
using Matrix = Physics.Walker.PPO.Matrix;
using Square = Physics.Objects.RigidBodies.Square;
using Water = Physics.Objects.SoftBodies.Water;

namespace Physics;

public class Game1 : Game
{
    private GraphicsDeviceManager _graphics;
    private readonly InputManager _input;
    private SpriteBatch _spriteBatch;
    private Renderer _renderer;
    
    private readonly List<IObject> _rigidObjects;
    private readonly List<IObject> _softObjects;
    private readonly List<Water> _water;
    private Walker.Walker _walker;

    private IMaterial _squareMaterial = new Wood();
    private readonly IMaterial _obstacleMaterial = new Metal();
    private const bool RoughFloor = false;

    private const int Iterations = 50;
    private int _speed = 1;

    private int EntityCount = 0;
    private int EntitySize = 50;
    private const int EntityCap = 200;
    
    private IObject _dragging;
    private Vector2[] _stringLine;

    private int Steps = 0;
    private Trajectory _currentTrajectory;
    private float _currentReward;
    
    private const string CriticFileLocation = "/Users/square/Projects/Physics/SavedModels/critic.txt";
    private const string ActorFileLocation = "/Users/square/Projects/Physics/SavedModels/actor.txt";
    
    public Game1()
    {
        _graphics = new GraphicsDeviceManager(this);
        _input = new InputManager();
        _rigidObjects = new List<IObject>();
        _currentTrajectory = new Trajectory();
        _softObjects = new List<IObject>();
        _water = new List<Water>();
        _stringLine = new Vector2[2];
        _dragging = null;
        _walker = new Walker.Walker();
        _currentReward = 0;
        
        _walker.CreateCreature(_rigidObjects);
        
        Content.RootDirectory = "Content";
        IsMouseVisible = true;
    }

    protected override void LoadContent()
    {
        _spriteBatch = new SpriteBatch(GraphicsDevice);
        _renderer = new Renderer(_spriteBatch);

        CreateFloor();
    }

    protected override void Initialize()
    {
        _graphics.IsFullScreen = false;
        _graphics.PreferredBackBufferWidth = 1000;
        _graphics.PreferredBackBufferHeight = 1000;
        _graphics.SynchronizeWithVerticalRetrace = false;
        IsFixedTimeStep = false;
        _graphics.ApplyChanges();
        
        base.Initialize();
    }
    
    private void CreateObstacles()
    {
        Vector2[] platformPositions = {
            new (100, 500), new (100, 450), new (400, 450), new (400, 500)
        };
        
        Vector2[] platform2Positions = {
            new (600, 500), new (600, 450), new (900, 450), new (900, 500)
        };

        Vector2[] platform3Positions = {
            new (350, 250), new (350, 200), new (650, 200), new (650, 250)
        };
        
        Hull platform = Hull.FromPositions(_obstacleMaterial, platformPositions, isStatic: true, isFloor: true);
        Hull platform2 = Hull.FromPositions(_obstacleMaterial, platform2Positions, isStatic: true, isFloor: true);
        Hull platform3 = Hull.FromPositions(_obstacleMaterial, platform3Positions, isStatic: true, isFloor: true);

        platform.Rotate(60);
        platform2.Rotate(-60);
        _rigidObjects.Add(platform);
        _rigidObjects.Add(platform2);
        _rigidObjects.Add(platform3);
    }

    private void CreateFloor()
    {
        if (RoughFloor)
        {
            CreateRoughFloor();
            return;
        }
        
        Vector2[] floorPositions = {
            new (-50, 1050), new (-50, 900), new (1050, 900), new (1050, 1050)
        };

        Hull floor = Hull.FromPositions(_obstacleMaterial, floorPositions, isStatic: true, isFloor: true);
        _rigidObjects.Add(floor);
        
    }

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
            
            Hull segment = Hull.FromPositions(_obstacleMaterial, positions, isStatic: true);
            _rigidObjects.Add(segment);
                
            previousVector = new Vector2(x, y);
        }
    }

    // perform action, then do physics update
    protected override void Update(GameTime gameTime)
    {
        base.Update(gameTime);
        _input.Update();
        
        float deltaTime = (float) gameTime.ElapsedGameTime.TotalSeconds;
        HandleInputs(deltaTime);
        UpdateEnvironment(deltaTime);

            //UpdateGUI();
    }   

    protected override void Draw(GameTime gameTime)
    {
        GraphicsDevice.Clear(Color.Black);
        RenderObjects();
        base.Draw(gameTime);
    }

    private void RenderObjects()
    {
        _spriteBatch.Begin();
        
        foreach (var iObject in _rigidObjects)
        {
            _renderer.RenderRigidObject(iObject);
        }
        
        _renderer.RenderJoint(_walker.GetJointColors());

        foreach (var iObject in _softObjects)
        {
            _renderer.RenderSoftObject(iObject);
        }
        
        foreach (var waterDrop in _water)
        {
            _renderer.RenderWater(waterDrop);
        }

        if (_stringLine != null)
        {
            _renderer.DrawLine(_stringLine[0], _stringLine[1], Color.LightGray, 1);
        }

        _spriteBatch.End();
    }

    private void StepObjects(float deltaTime)
    {
        deltaTime /= Iterations;
        
        for (int i = 0; i < Iterations; i++)
        {
            foreach (var joint in _walker.GetJoints())
            {
                joint.Step();
            }
            

            foreach (var rigidObject in _rigidObjects)
            {
                rigidObject.Update(_rigidObjects, _softObjects, deltaTime);
            };

            foreach (var softObject in _softObjects)
            {
                softObject.Update(_rigidObjects, _softObjects, deltaTime);
            }

            foreach (var water in _water)
            {
                water.Update(deltaTime, _rigidObjects, _water);
            };
        }
    }

    private void HandleInputs(float deltaTime)
    {
        if (EntityCount < EntityCap)
        {
            if (_input.IsKeyPressed(Keys.S)) 
            {
                Square square = Square.FromSize(_squareMaterial, _input.GetMousePosition(), EntitySize);
                square.AddAcceleration(new Vector2(0, 980f)); // gravity
                //square.AddAcceleration(new Vector2(200f, 0)); // wind
                EntityCount += 1;
                _rigidObjects.Add(square);
            }
            
            if (_input.IsKeyPressed(Keys.P)) 
            {
                Triangle triangle = Triangle.FromSize(_squareMaterial, _input.GetMousePosition(), EntitySize);
                triangle.AddAcceleration(new Vector2(0, 980f));
                EntityCount += 1;
                _rigidObjects.Add(triangle);
            }
            
            if (_input.IsKeyPressed(Keys.O)) 
            {
                Pole pole = Pole.FromSize(_squareMaterial, _input.GetMousePosition(), EntitySize);
                pole.AddAcceleration(new Vector2(0, 980f));
                EntityCount += 1;
                _rigidObjects.Add(pole);
            }
        
            if (_input.IsKeyPressed(Keys.C)) 
            {
                Square square = Square.FromSize(_squareMaterial, _input.GetMousePosition(), EntitySize);
                square.AddAcceleration(new Vector2(0, 980f));
                square.SmoothCorners(2);
                EntityCount += 1;
                _rigidObjects.Add(square);
            }
        
            if (_input.IsKeyPressed(Keys.H)) 
            {
                Hexagon hexagon = Hexagon.FromSize(_squareMaterial, _input.GetMousePosition(), EntitySize);
                hexagon.AddAcceleration(new Vector2(0, 980f));
                EntityCount += 1;
                _rigidObjects.Add(hexagon);
            }

            if (_input.IsKeyPressed(Keys.D)) 
            {
                Objects.SoftBodies.Square square = Objects.SoftBodies.Square.FromSize(_squareMaterial, _input.GetMousePosition(), EntitySize / 10, EntitySize / 5);
                square.AddAcceleration(new Vector2(0, 980));
                _softObjects.Add(square);
                EntityCount += 10; 
            }
        }
        
        if (_input.IsKeyPressed(Keys.Q))
        {
            StartTraining();
        }
        
        if (_input.IsKeyPressed(Keys.I))
        {
            _squareMaterial = new Ice();
        }
        
        if (_input.IsKeyPressed(Keys.T))
        {
            _squareMaterial = new Titanium();
        }

        if (_input.IsKeyPressed(Keys.W))
        {
            _squareMaterial = new Wood();
        }

        if (_input.IsKeyPressed(Keys.R))
        {
            _squareMaterial = new Rubber();
        }
        
        if (_input.IsKeyPressed(Keys.J))
        {
            _squareMaterial = new Carpet();
        }
        
        if (_input.IsKeyPressed(Keys.K))
        {
            Console.WriteLine("Saving the current weights.");
            _walker.Save(CriticFileLocation, ActorFileLocation);
        }
        
        if (_input.IsKeyPressed(Keys.L))
        {
            Console.WriteLine("Loading the current weights.");
            _walker.Load(CriticFileLocation, ActorFileLocation);
            Reset();
        }

        if (_input.IsKeyPressed(Keys.Up))
        {
            EntitySize += 10;
        }
        
        if (_input.IsKeyPressed(Keys.Down))
        {
            if (EntitySize >= 20)
            {
                EntitySize -= 10;
            }
        }

        if (_input.IsKeyPressed(Keys.Right))
        {
            _speed += 1;
            Console.WriteLine($"Speed is now {_speed}.");
        }

        if (_input.IsKeyPressed(Keys.Left))
        {
            _speed -= 1;
            if (_speed < 1) _speed = 1;
            Console.WriteLine($"Speed is now {_speed}.");
        }
        
        if (_input.IsMouseHeld())
        {
            _dragging ??= GetDraggedObject(_input.GetMousePosition());
            if (_dragging == null) return;
            
            RigidBody iBody = (RigidBody)_dragging;
            if (iBody.IsStatic()) return;

            Vector2 normal = (_input.GetMousePosition() - iBody.GetCentroid());
            Vector2 linearVelocity = normal * deltaTime * 10;
            iBody.AddLinearVelocity(linearVelocity);

            _stringLine = new[] { _input.GetMousePosition(), iBody.GetCentroid() };
        }
        else
        {
            _dragging = null;
            _stringLine = null;
        }
    }

    private IObject GetDraggedObject(Vector2 mousePos)
    {
        foreach (var rigidObject in _rigidObjects)
        {
            RigidBody iBody = (RigidBody) rigidObject;
            if (RayCasting.IsColliding(mousePos, iBody.GetVectors()))
            {
                return rigidObject;
            }
        }

        return null;
    }

    private void UpdateEnvironment(float deltaTime)
    {
        for (int i = 0; i < _speed; i++)
        {
            Steps += 1;
            _walker.Update();

            if (Steps != 1) TakeAction();
            StepObjects(deltaTime);
            CheckState();
        }
    }

    private void TakeAction()
    {
        Matrix state = _walker.GetState();
        _currentTrajectory.States.Add(state);
        
        _walker.GetAction(state, out Matrix probabilities);
        _currentTrajectory.Probabilities.Add(probabilities);
    }

    private void CheckState()
    {
        GetReward();

        if (_walker.GetPosition().X > 850)
        {
            _currentReward += 1000;
            TerminalState();
            return;
        }

        if (_walker.Terminal || Steps >= 4000)
        {
            _currentReward -= 100;
            TerminalState();
            return;
        }
        
        _currentTrajectory.Rewards.Add(_currentReward);
    }

    private void TerminalState()
    {
        _currentTrajectory.Rewards.Add(_currentReward);
        _walker.RepairBody();
        _walker.Terminal = false;

        StartTraining();
    }

    // do position set by user?
    private void GetReward()
    {
        _currentReward = _walker.GetChangeInPosition().X;
        if (_walker.GetPosition().Y > 850)
        {
            _currentReward -= 0.01f;
        }
    }
    
    private void StartTraining()
    {
        Reset();
        TrainNetworks();
        
        _currentTrajectory = new Trajectory();
    }

    private void TrainNetworks()
    {
        _walker.Train(_currentTrajectory);
    }

    private void Reset()
    {
        _walker.Reset();
        _rigidObjects.Clear();
        _softObjects.Clear();
        _water.Clear();
        EntityCount = 0;
        Steps = 0;
        CreateFloor();
        _walker.CreateCreature(_rigidObjects);
    }
}