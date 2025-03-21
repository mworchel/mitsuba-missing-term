import drjit as dr
import mitsuba as mi

def compute_gradient_finite_differences(func: callable, x: float, h: float = 0.01):
    h = dr.opaque(mi.Float, h)
    return (func(x + h) - func(x - h)) / (2*h)

def compute_gradient_forward(func: callable, x: float):
    x_attached = mi.Float(x)
    dr.enable_grad(x_attached)

    # TODO: It is recommended to forward right before mi.render
    output = func(x_attached)

    dr.set_grad(x_attached, 1)
    dr.forward_from(x_attached)
    return dr.grad(output)