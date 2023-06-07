namespace Physics.Materials;

public interface IMaterial
{
    public float Density { get; set;  }
    public float Restitution { get; set; }
    public bool Static { get; set; }
}