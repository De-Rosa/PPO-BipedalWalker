using System.Collections.Generic;
using System;
using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using NEA.Bodies;

namespace NEA.Rendering;

// Renderer class, allows for drawing of rigid bodies and console rendering.
public sealed class Renderer
{
    private readonly SpriteBatch _spriteBatch;
    private readonly Texture2D _lineTexture;
    private readonly ConsoleRenderer _consoleRenderer;

    private Vector2 _camera;
    
    public Renderer(SpriteBatch spriteBatch)
    {
        _spriteBatch = spriteBatch;
        _lineTexture = new Texture2D(spriteBatch.GraphicsDevice, 1, 1);
        _lineTexture.SetData(new[] { Color.White });

        _consoleRenderer = new ConsoleRenderer();
        
        _camera = Vector2.Zero;
    }

    // Updates the console
    public void UpdateConsole()
    {
        _consoleRenderer.Update();
    }

    // Updates the console, rendering the current episode information.
    public void UpdateConsole(int episode, int timeStep, float distance, float averageReward, float bestDistance, float pastAverageReward, Walker.PPO.Matrix state)
    {
        _consoleRenderer.Update(episode, timeStep, distance, averageReward, bestDistance, pastAverageReward, state);
    }

    // Updates the console, rendering the current training information.
    public void UpdateConsole(int epoch, int batch, int batchSize, float criticLoss)
    {
        _consoleRenderer.Update(epoch, batch, batchSize, criticLoss);
    }

    // Render the exit menu, occuring when 'x' is pressed during an episode.
    public void ExitTraining()
    {
        _consoleRenderer.ExitTraining();
    }
    
    // Adds the average episode reward to the list for use in data collection.
    public void AddAverageEpisodeReward(float reward)
    {
        _consoleRenderer.AddAverageEpisodeReward(reward);
    }

    // Moves the camera by a vector.
    public void MoveCamera(Vector2 vector)
    {
        _camera += vector;
    }

    // Resets the camera position.
    public void ResetCamera()
    {
        _camera = Vector2.Zero;
    }

    // Returns the position of the camera.
    public Vector2 GetCameraPosition()
    {
        return _camera;
    }

    // Renders a rigid object by drawing line between its vertices.
    public void RenderRigidObject(RigidBody body)
    {
        List<Vector2> vectors = body.GetVectors();
        Color color = body.IsStatic() ? Color.White : body.GetColor();
        
        for (int i = 0; i < vectors.Count; i++)
        {
            DrawLine(vectors[i] + _camera, vectors[(i+1) % vectors.Count] + _camera, color);
        }
    }

    // Renders the joints of a creature by drawing a coloured square at each position.
    public void RenderJoints(List<Tuple<Vector2, Color>> colors)
    {
        foreach (var color in colors)
        {
            DrawSquare(color.Item1, 3, color.Item2, 2f);
        }
    }

    // Draws a line between two points.
    public void DrawLine(Vector2 point1, Vector2 point2, Color color, float thickness = 1f)
    {
        var distance = Vector2.Distance(point1, point2);
        var angle = (float)Math.Atan2(point2.Y - point1.Y, point2.X - point1.X);
        DrawLine(point1 + _camera, distance, angle, color, thickness);
    }

    // Draws a line from a given position and angle.
    private void DrawLine(Vector2 point, float length, float angle, Color color, float thickness = 1f)
    {
        var origin = new Vector2(0f, 0.5f);
        var scale = new Vector2(length, thickness);
        _spriteBatch.Draw(_lineTexture, point + _camera, null, color, angle, origin, scale, SpriteEffects.None, 0);
    }

    // Draws a square at a point.
    public void DrawSquare(Vector2 origin, float length, Color color, float thickness = 1f)
    {
        var squareOrigin = new Vector2(origin.X - length / 2, origin.Y - length / 2);
        Vector2[] points = new Vector2[] { squareOrigin, new Vector2(squareOrigin.X + length, squareOrigin.Y), new Vector2(squareOrigin.X + length, squareOrigin.Y + length), new Vector2(squareOrigin.X, squareOrigin.Y + length)};
        for (int i = 0; i < points.Length; i++)
        {
            DrawLine(points[i] + _camera, points[(i + 1) % points.Length] + _camera, color, thickness);
        }
    }
}