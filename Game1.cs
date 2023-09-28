using System;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Physics.Input;
using Physics.Rendering;

namespace Physics;

public class Game1 : Game
{
    private GraphicsDeviceManager _graphics;
    private readonly InputManager _input;
    private SpriteBatch _spriteBatch;
    private Renderer _renderer;
    private Environment _environment;
    
    public Game1()
    {
        _graphics = new GraphicsDeviceManager(this);
        _input = new InputManager();
     
        Content.RootDirectory = "Content";
        IsMouseVisible = true;
    }

    protected override void LoadContent()
    {
        _spriteBatch = new SpriteBatch(GraphicsDevice);
        _renderer = new Renderer(_spriteBatch);

        _renderer.UpdateConsole();
        
        _environment = new Environment(_spriteBatch, _renderer);
        _environment.InitialState();
    }

    protected override void Initialize()
    {
        _graphics.IsFullScreen = false;
        _graphics.PreferredBackBufferWidth = 1000;
        _graphics.PreferredBackBufferHeight = 1000;
        IsFixedTimeStep = true;
        _graphics.ApplyChanges();

        base.Initialize();
    }

    protected override void Update(GameTime gameTime)
    {
        float deltaTime = (float) gameTime.ElapsedGameTime.TotalSeconds;
    
        base.Update(gameTime);
        _input.Update();

        (int episode, int timeStep, float distance, float averageReward, float bestDistance, float pastAverageReward) =
            _environment.GetConsoleInformation();
        _renderer.UpdateConsole(episode, timeStep, distance, averageReward, bestDistance, pastAverageReward);
    
    
        _environment.Update(deltaTime);
        HandleInputs(deltaTime);
        
    }   

    protected override void Draw(GameTime gameTime)
    {
        GraphicsDevice.Clear(Color.Black);
        _environment.RenderObjects();
        base.Draw(gameTime);
    }
    
    private void HandleInputs(float deltaTime)
    {
        if (_input.IsKeyPressed(Keys.X))
        {
            _renderer.ExitTraining();    
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
    }
}