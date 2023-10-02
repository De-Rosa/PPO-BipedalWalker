using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;

namespace Physics.Input;

// Input Manager class, allows for key input inside the GUI.
public class InputManager
{
    private KeyboardState _currentKeyboardState;
    private KeyboardState _lastKeyboardState;

    private MouseState _currentMouseState;
    private MouseState _lastMouseState;
    
    public void Update()
    {
        _lastKeyboardState = _currentKeyboardState;
        _lastMouseState = _currentMouseState;

        _currentKeyboardState = Keyboard.GetState();
        _currentMouseState = Mouse.GetState();
    }

    public bool IsKeyPressed(Keys key)
    {
        return _currentKeyboardState.IsKeyDown(key) && _lastKeyboardState.IsKeyUp(key);
    }

    public bool IsKeyHeld(Keys key)
    {
        return _currentKeyboardState.IsKeyDown(key) && _lastKeyboardState.IsKeyDown(key);
    }

    public Vector2 GetMousePosition()
    {
        return new Vector2(_currentMouseState.X, _currentMouseState.Y); 
    }
    
    public Vector2 GetPreviousMousePosition()
    {
        return new Vector2(_lastMouseState.X, _lastMouseState.Y); 
    }

    public bool IsMousePressed()
    {
        return _currentMouseState.LeftButton == ButtonState.Pressed && _lastMouseState.LeftButton == ButtonState.Released;
    }

    public bool IsMouseHeld()
    {
        return _currentMouseState.LeftButton == ButtonState.Pressed && _lastMouseState.LeftButton == ButtonState.Pressed;
    }
}