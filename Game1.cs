using System;
using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Physics.Bodies;
using Physics.Bodies.Physics;
using Physics.Input;
using Physics.Materials;
using Physics.Objects;
using Physics.Objects.RigidBodies;
using Physics.Rendering;
using Square = Physics.Objects.RigidBodies.Square;

namespace Physics;

public class Game1 : Game
{
    private GraphicsDeviceManager _graphics;
    private readonly InputManager _input;
    private SpriteBatch _spriteBatch;
    private Renderer _renderer;
    
    private readonly List<RigidBody> _rigidObjects;
    private readonly List<Water> _water;
    private readonly Environment _environment;

    private IMaterial _squareMaterial = new Wood();

    private const int Iterations = 50;

    private int _entityCount = 0;
    private int _entitySize = 50;
    private const int EntityCap = 200;
    
    private RigidBody _dragging;
    private Vector2[] _stringLine;

    public Game1()
    {
        _graphics = new GraphicsDeviceManager(this);
        _input = new InputManager();
        _rigidObjects = new List<RigidBody>();
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

        base.Initialize();
    }

    protected override void Update(GameTime gameTime)
    {
        base.Update(gameTime);
        _input.Update();
        
        float deltaTime = (float) gameTime.ElapsedGameTime.TotalSeconds;
        _environment.Update();
        StepObjects(deltaTime);
        _environment.UpdateReward(_rigidObjects);
        
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
        
        _environment.RenderWalker(_renderer);
        
        foreach (var iObject in _rigidObjects)
        {
            _renderer.RenderRigidObject(iObject);
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
            _environment.StepWalker(deltaTime);
            
            foreach (var body in _rigidObjects)
            {
                var rigidObject = (IObject) body;
                rigidObject.Update(_rigidObjects, deltaTime);
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

        if (_input.IsKeyHeld(Keys.Right))
        {
            _renderer.MoveCamera(new Vector2(-1, 0));
        }

        if (_input.IsKeyHeld(Keys.Left))
        {
            _renderer.MoveCamera(new Vector2(1, 0));
        }
        
        if (_input.IsKeyHeld(Keys.Up))
        {
            _renderer.MoveCamera(new Vector2(0, 1));
        }

        if (_input.IsKeyHeld(Keys.Down))
        {
            _renderer.MoveCamera(new Vector2(0, -1));
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

    private RigidBody GetDraggedObject(Vector2 mousePos)
    {
        foreach (var rigidObject in _rigidObjects)
        {
            if (RayCasting.IsColliding(mousePos, rigidObject.GetVectors()))
            {
                return rigidObject;
            }
        }

        return null;
    }
}