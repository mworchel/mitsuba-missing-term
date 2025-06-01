import drjit as dr
import mitsuba as mi

def det_over_det(D):
    return dr.select(D != 0, dr.replace_grad(1, D/dr.detach(D)), 0)

def solid_to_surface_reparam_det(si: mi.SurfaceInteraction3f, x_prev: mi.Point3f):
    """ Reparameterization determinant from solid angles to surface elements
    """

    d = si.p - x_prev

    distance_squared = dr.squared_norm(d)
    cos_theta = dr.abs_dot(si.n, -dr.normalize(d))

    # If the intersection point lies on an environment emitter, 
    # the surface parameterization is invalid (and si.is_valid() == False)
    det = dr.select(si.is_valid() & (distance_squared > 0), 
                    dr.norm(dr.cross(si.dp_du, si.dp_dv)) * dr.abs(cos_theta) / distance_squared, 1)

    return det

def sensor_to_solid_reparam_det(sensor: mi.Sensor, si: mi.SurfaceInteraction3f):
    """ Reparameterization determinant from sensor elements to solid angles 
        (for perspective cameras)
    """
    sensor_pos = sensor.world_transform() @ mi.Point3f(0)
    sensor_dir = sensor.world_transform() @ mi.Vector3f(0, 0, 1)
    
    near_clip = sensor.near_clip()

    # Jacobian determinant (sensor to solid angle)
    cos_phi = dr.abs_dot(dr.normalize(si.p - sensor_pos), sensor_dir)
    return dr.select(si.is_valid(), dr.square(near_clip) / cos_phi*cos_phi*cos_phi, 1.)
