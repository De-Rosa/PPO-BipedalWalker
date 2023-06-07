using System.Collections.Generic;
using Microsoft.Xna.Framework;
using Physics.Bodies;

namespace Physics.Objects;

public interface IObject
{ 
    public void Update(List<IObject> rigidBodies, List<IObject> softBodies, float deltaTime);
    public IBody GetBody();
}