using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Input;

namespace Physics.Input;

public class InputManager
{
    private KeyboardState _currentKeyboardState = new KeyboardState();
    private KeyboardState _lastKeyboardState = new KeyboardState();

    private MouseState _currentMouseState = new MouseState();
    private MouseState _lastMouseState = new MouseState();
    
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