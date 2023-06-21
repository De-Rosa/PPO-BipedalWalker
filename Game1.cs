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
    private readonly Environment _environment;

    private IMaterial _squareMaterial = new Wood();

    private const int Iterations = 50;
    private int _speed = 1;

    private int _entityCount = 0;
    private int _entitySize = 50;
    private const int EntityCap = 200;
    
    private IObject _dragging;
    private Vector2[] _stringLine;

    public Game1()
    {
        _graphics = new GraphicsDeviceManager(this);
        _input = new InputManager();
        _rigidObjects = new List<IObject>();
        _softObjects = new List<IObject>();
        _water = new List<Water>();
        _stringLine = new Vector2[2];
        _dragging = null;
        _environment = new Environment(_rigidObjects);
        
        Content.RootDirectory = "Content";
        IsMouseVisible = true;
    }

    protected override void LoadContent()
    {
        _spriteBatch = new SpriteBatch(GraphicsDevice);
        _renderer = new Renderer(_spriteBatch);
    }

    protected override void Initialize()
    {
        _graphics.IsFullScreen = false;
        _graphics.PreferredBackBufferWidth = 1000;
        _graphics.PreferredBackBufferHeight = 1000;
        _graphics.SynchronizeWithVerticalRetrace = false;
        IsFixedTimeStep = false;
        _graphics.ApplyChanges();
        
        _environment.CreateCreatures(_rigidObjects);

        base.Initialize();
    }

    // perform action, then do physics update
    protected override void Update(GameTime gameTime)
    {
        base.Update(gameTime);
        _input.Update();
        
        float deltaTime = (float) gameTime.ElapsedGameTime.TotalSeconds;
        
        _environment.UpdateWalkers();
        _environment.TakeActions();
        StepObjects(deltaTime);
        _environment.CheckStates(_rigidObjects);

        HandleInputs(deltaTime);
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
        
        _environment.RenderWalkers(_renderer);

        foreach (var iObject in _rigidObjects)
        {
            _renderer.RenderRigidObject(iObject);
        }
        
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
            _environment.StepWalkers(deltaTime);

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
        if (_entityCount < EntityCap)
        {
            if (_input.IsKeyPressed(Keys.S)) 
            {
                Square square = Square.FromSize(_squareMaterial, _input.GetMousePosition(), _entitySize);
                square.AddAcceleration(new Vector2(0, 980f)); // gravity
                //square.AddAcceleration(new Vector2(200f, 0)); // wind
                _entityCount += 1;
                _rigidObjects.Add(square);
            }
            
            if (_input.IsKeyPressed(Keys.P)) 
            {
                Triangle triangle = Triangle.FromSize(_squareMaterial, _input.GetMousePosition(), _entitySize);
                triangle.AddAcceleration(new Vector2(0, 980f));
                _entityCount += 1;
                _rigidObjects.Add(triangle);
            }

            if (_input.IsKeyPressed(Keys.O)) 
            {
                Pole pole = Pole.FromSize(_squareMaterial, _input.GetMousePosition(), _entitySize);
                pole.AddAcceleration(new Vector2(0, 980f));
                _entityCount += 1;
                _rigidObjects.Add(pole);
            }
        
            if (_input.IsKeyPressed(Keys.C)) 
            {
                Square square = Square.FromSize(_squareMaterial, _input.GetMousePosition(), _entitySize);
                square.AddAcceleration(new Vector2(0, 980f));
                square.SmoothCorners(2);
                _entityCount += 1;
                _rigidObjects.Add(square);
            }
        
            if (_input.IsKeyPressed(Keys.H)) 
            {
                Hexagon hexagon = Hexagon.FromSize(_squareMaterial, _input.GetMousePosition(), _entitySize);
                hexagon.AddAcceleration(new Vector2(0, 980f));
                _entityCount += 1;
                _rigidObjects.Add(hexagon);
            }

            if (_input.IsKeyPressed(Keys.D)) 
            {
                Objects.SoftBodies.Square square = Objects.SoftBodies.Square.FromSize(_squareMaterial, _input.GetMousePosition(), _entitySize / 10, _entitySize / 5);
                square.AddAcceleration(new Vector2(0, 980));
                _softObjects.Add(square);
                _entityCount += 10; 
            }
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
            _environment.Save();
        }
        
        if (_input.IsKeyPressed(Keys.L))
        {
            Console.WriteLine("Loading the current weights.");
            _environment.Load();
        }

        if (_input.IsKeyPressed(Keys.Down))
        {
            if (_entitySize >= 20)
            {
                _entitySize -= 10;
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
}