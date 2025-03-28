from __future__ import annotations # Delayed parsing of type annotations

import drjit as dr
import mitsuba as mi

from mitsuba.python.ad.integrators.common import ADIntegrator, RBIntegrator, PSIntegrator, mis_weight
from mitsuba.python.ad.integrators.prb_projective import PathProjectiveIntegrator

from .prb_threepoint import PRBThreePointIntegrator

class PathProjectiveFixIntegrator(PathProjectiveIntegrator):
    r"""
    .. _integrator-prb_projective:

    Projective sampling Path Replay Backpropagation (PRB) (:monosp:`prb_projective`)
    --------------------------------------------------------------------------------

    .. pluginparameters::

     * - max_depth
       - |int|
       - Specifies the longest path depth in the generated output image (where -1
         corresponds to :math:`\infty`). A value of 1 will only render directly
         visible light sources. 2 will lead to single-bounce (direct-only)
         illumination, and so on. (Default: -1)

     * - rr_depth
       - |int|
       - Specifies the path depth, at which the implementation will begin to use
         the *russian roulette* path termination criterion. For example, if set to
         1, then path generation many randomly cease after encountering directly
         visible surfaces. (Default: 5)

     * - sppc
       - |int|
       - Number of samples per pixel used to estimate the continuous
         derivatives. This is overriden by whatever runtime `spp` value is
         passed to the `render()` method. If this value was not set and no
         runtime value `spp` is used, the `sample_count` of the film's sampler
         will be used.

     * - sppp
       - |int|
       - Number of samples per pixel used to to estimate the gradients resulting
         from primary visibility changes (on the first segment of the light
         path: from the sensor to the first bounce) derivatives. This is
         overriden by whatever runtime `spp` value is passed to the `render()`
         method. If this value was not set and no runtime value `spp` is used,
         the `sample_count` of the film's sampler will be used.

     * - sppi
       - |int|
       - Number of samples per pixel used to to estimate the gradients resulting
         from indirect visibility changes  derivatives. This is overriden by
         whatever runtime `spp` value is passed to the `render()` method. If
         this value was not set and no runtime value `spp` is used, the
         `sample_count` of the film's sampler will be used.

     * - guiding
       - |string|
       - Guiding type, must be one of: "none", "grid", or "octree". This
         specifies the guiding method used for indirectly observed
         discontinuities. (Default: "octree")

     * - guiding_proj
       - |bool|
       - Whether or not to use projective sampling to generate the set of
         samples that are used to build the guiding structure. (Default: True)

     * - guiding_rounds
       - |int|
       - Number of sampling iterations used to build the guiding data structure.
         A higher number of rounds will use more samples and hence should result
         in a more accurate guiding structure. (Default: 1)

    This plugin implements a projective sampling Path Replay Backpropagation
    (PRB) integrator with the following features:

    - Emitter sampling (a.k.a. next event estimation).

    - Russian Roulette stopping criterion.

    - Projective sampling. This means that it can handle discontinuous
      visibility changes, such as a moving shape's gradients.

    - Detached sampling. This means that the properties of ideal specular
      objects (e.g., the IOR of a glass vase) cannot be optimized.

    In order to estimate the indirect visibility discontinuous derivatives, this
    integrator starts by sampling a boundary segment and then attempts to
    connect it to the sensor and an emitter. It is effectively building lights
    paths from the middle outwards. In order to stay within the specified
    `max_depth`, the integrator starts by sampling a path to the sensor by using
    reservoir sampling to decide whether or not to use a longer path. Once a
    path to the sensor is found, the other half of the full light path is
    sampled.

    See the paper :cite:`Zhang2023Projective` for details on projective
    sampling, and guiding structures for indirect visibility discontinuities.

    .. warning::
        This integrator is not supported in variants which track polarization
        states.

    .. tabs::

        .. code-tab:: python

            'type': 'prb_projective',
            'sppc': 32,
            'sppp': 32,
            'sppi': 128,
            'guiding': 'octree',
            'guiding_proj': True,
            'guiding_rounds': 1
    """

    def __init__(self, props):
        super().__init__(props)

        # Override the radiative backpropagation flag to allow the parent class
        # to call the sample() method following the logics defined in the
        # ``PRBThreePointIntegrator``
        self.radiative_backprop = True

        # Specify the seed ray generation strategy
        self.project_seed = props.get('project_seed', 'both')
        if self.project_seed not in ['both', 'bsdf', 'emitter']:
            raise Exception(f"Project seed must be one of 'both', 'bsdf', "
                            f"'emitter', got '{self.project_seed}'")

    def RB_render_forward(self: mi.SamplingIntegrator,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: mi.UInt32 = 0,
                       spp: int = 0) -> mi.TensorXf:

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        
        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, self.aov_names())

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            ray, weight, pos = self.sample_rays(scene, sensor, sampler)


            # Launch the Monte Carlo sampling process in primal mode (1)
            L, valid, aovs, state_out = PRBThreePointIntegrator.sample(self,
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler.clone(),
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                state_in=None,
                active=mi.Bool(True)
            )
            
            # Launch the Monte Carlo sampling process in forward mode (2)
            δL, valid_2, δaovs, state_out_2 = PRBThreePointIntegrator.sample(self,
                mode=dr.ADMode.Forward,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                δaovs=None,
                state_in=state_out,
                active=mi.Bool(True)
            )

            with dr.resume_grad():
                
                block = film.create_block()
                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)
                si = scene.ray_intersect(ray,
                                         ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                         coherent=mi.Bool(True))
                pos = dr.select(valid, sensor.sample_direction(si, [0, 0], active=valid)[0].uv, pos)
                diff = si.p-ray.o
                dist_squared = dr.squared_norm(diff)
                dp = dr.dot(dr.normalize(diff), si.n)
                D = dr.select(valid, dr.norm(dr.cross(si.dp_du, si.dp_dv)) * -dp / dist_squared , 1.)
                #Accumulate into the image block
                ADIntegrator._splat_to_block(
                    block, film, pos,
                    value=L * weight * dr.replace_grad(1, D/dr.detach(D)),
                    weight=dr.replace_grad(1, D/dr.detach(D)),
                    alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                    aovs=aovs,
                    wavelengths=ray.wavelengths
                )
                # Perform the weight division
                film.put_block(block)
                
                result_img = film.develop()

                # Propagate the gradients to the image tensor
                dr.forward_to(result_img, flags=dr.ADFlag.ClearNone | dr.ADFlag.AllowNoGrad)
                first_hit = dr.grad(result_img)

            film.clear()

            # Prepare an ImageBlock as specified by the film
            block = film.create_block()

            # Only use the coalescing feature when rendering enough samples
            block.set_coalesce(block.coalesce() and spp >= 4)

            # Accumulate into the image block
            ADIntegrator._splat_to_block(
                block, film, pos,
                value=δL * weight,
                weight=1.0,
                alpha=dr.select(valid_2, mi.Float(1), mi.Float(0)),
                aovs=[δaov * weight for δaov in δaovs],
                wavelengths=ray.wavelengths
            )
            
            # Perform the weight division and return an image tensor
            film.put_block(block)
            result_grad = film.develop()


        # Explicitly delete any remaining unused variables
        del sampler, ray, weight, pos, L, valid, aovs, δL, δaovs, \
            valid_2, params, state_out, state_out_2#, block
        
        return result_grad + first_hit
    
    def RB_render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: mi.UInt32 = 0,
                        spp: int = 0) -> None:
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(sensor, seed, spp, self.aov_names())

            # Generate a set of rays starting at the sensor, keep track of
            # derivatives wrt. sample positions ('pos') if there are any
            ray, weight, pos = self.sample_rays(scene, sensor, sampler)

            def splatting_and_backward_gradient_image(value: mi.Spectrum,
                                                      weight: mi.Float,
                                                      alpha: mi.Float,
                                                      aovs: Sequence[mi.Float]):
                '''
                Backward propagation of the gradient image through the sample
                splatting and weight division steps.
                '''

                # Prepare an ImageBlock as specified by the film
                block = film.create_block()

                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)

                ADIntegrator._splat_to_block(
                    block, film, pos,
                    value=value,
                    weight=weight,
                    alpha=alpha,
                    aovs=aovs,
                    wavelengths=ray.wavelengths
                )

                film.put_block(block)

                image = film.develop()

                dr.set_grad(image, grad_in)
                dr.enqueue(dr.ADMode.Backward, image)
                dr.traverse(dr.ADMode.Backward)

            # Differentiate sample splatting and weight division steps to
            # retrieve the adjoint radiance (e.g. 'δL')
            with dr.resume_grad():
                with dr.suspend_grad(pos, ray, weight):
                    L = dr.full(mi.Spectrum, 1.0, dr.width(ray))
                    dr.enable_grad(L)
                    aovs = []
                    for _ in self.aov_names():
                        aov = dr.ones(mi.Float, dr.width(ray))
                        dr.enable_grad(aov)
                        aovs.append(aov)
                    splatting_and_backward_gradient_image(
                        value=L * weight,
                        weight=1.0,
                        alpha=1.0,
                        aovs=[aov * weight for aov in aovs]
                    )

                    δL = dr.grad(L)
                    δaovs = dr.grad(aovs)

            # Clear the dummy data splatted on the film above
            film.clear()

            # Launch the Monte Carlo sampling process in primal mode (1)
            L, valid, aovs, state_out = PRBThreePointIntegrator.sample(self,
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler.clone(),
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                δaovs=None,
                state_in=None,
                active=mi.Bool(True)
            )

            # Launch Monte Carlo sampling in backward AD mode (2)
            L_2, valid_2, aovs_2, state_out_2 = PRBThreePointIntegrator.sample(self,
                mode=dr.ADMode.Backward,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=δL,
                δaovs=δaovs,
                state_in=state_out,
                active=mi.Bool(True)
            )

            with dr.resume_grad():
                si = scene.ray_intersect(ray,
                                         ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                         coherent=mi.Bool(True))
                pos = dr.select(valid, sensor.sample_direction(si, [0, 0], active=valid)[0].uv, pos)
                diff = si.p-ray.o
                dist_squared = dr.squared_norm(diff)
                dp = dr.dot(dr.normalize(diff), si.n)
                D = dr.select(valid, dr.norm(dr.cross(si.dp_du, si.dp_dv)) * -dp / dist_squared , 1.)


                # Prepare an ImageBlock as specified by the film
                block2 = film.create_block()

                # Only use the coalescing feature when rendering enough samples
                block2.set_coalesce(block2.coalesce() and spp >= 4)

                # Accumulate into the image block
                ADIntegrator._splat_to_block(
                    block2, film, pos,
                    value=L * weight * dr.replace_grad(1, D/dr.detach(D)),
                    weight=dr.replace_grad(1, D/dr.detach(D)),
                    alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                    aovs=aovs,
                    wavelengths=ray.wavelengths
                )

                # film the weight division and return an image tensor
                film.put_block(block2)
                image = film.develop()
                dr.set_grad(image, grad_in)

                dr.enqueue(dr.ADMode.Backward, image)
                dr.traverse(dr.ADMode.Backward)
                
                dr.eval()
                
            # We don't need any of the outputs here
            del L_2, valid_2, aovs_2, state_out, state_out_2, \
                δL, δaovs, ray, weight, pos, sampler


            # Run kernel representing side effects of the above
            dr.eval()

    def render_forward(self,
                       scene: mi.Scene,
                       params: Any,
                       sensor: Union[int, mi.Sensor] = 0,
                       seed: mi.UInt32 = 0,
                       spp: int = 0) -> mi.TensorXf:
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        shape = (film.crop_size()[1],
                 film.crop_size()[0],
                 film.base_channels_count() + len(self.aov_names()))
        result_grad = dr.zeros(mi.TensorXf, shape=shape)

        sampler_spp = sensor.sampler().sample_count()
        sppc = self.override_spp(self.sppc, spp, sampler_spp)
        sppp = self.override_spp(self.sppp, spp, sampler_spp)
        sppi = self.override_spp(self.sppi, spp, sampler_spp)
        
        # Continuous derivative (if RB is used)
        if self.radiative_backprop and sppc > 0:
            result_grad += self.RB_render_forward(scene, None, sensor, seed, sppc)

        # Discontinuous derivative (and the non-RB continuous derivative)
        if sppp > 0 or sppi > 0 or \
           (sppc > 0 and not self.radiative_backprop):

            # Compute an image with all derivatives attached
            ad_img = self.render_ad(scene, sensor, seed, spp, dr.ADMode.Forward)

            # We should only complain about the parameters not being attached
            # if `ad_img` isn't attached and we haven't used RB for the
            # continuous derivatives.
            if dr.grad_enabled(ad_img) or not self.radiative_backprop:
                dr.forward_to(ad_img)
                grad_img = dr.grad(ad_img)
                result_grad += grad_img
        
        return result_grad

    def render_backward(self,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: mi.UInt32 = 0,
                        spp: int = 0) -> None:
        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        sampler_spp = sensor.sampler().sample_count()
        sppc = self.override_spp(self.sppc, spp, sampler_spp)
        sppp = self.override_spp(self.sppp, spp, sampler_spp)
        sppi = self.override_spp(self.sppi, spp, sampler_spp)

        # Continuous derivative (if RB is used)
        if self.radiative_backprop and sppc > 0:
            RBIntegrator.render_backward(self, scene, None, grad_in, sensor, seed, sppc)

        # Discontinuous derivative (and the non-RB continuous derivative)
        if sppp > 0 or sppi > 0 or \
           (sppc > 0 and not self.radiative_backprop):

            # Compute an image with all derivatives attached
            ad_img = self.render_ad(
                scene, sensor, seed, spp, dr.ADMode.Backward)

            dr.set_grad(ad_img, grad_in)
            dr.enqueue(dr.ADMode.Backward, ad_img)
            dr.traverse(dr.ADMode.Backward)

        dr.eval()

    # @dr.syntax
    # def sample(self,
    #            mode: dr.ADMode,
    #            scene: mi.Scene,
    #            sampler: mi.Sampler,
    #            ray: mi.Ray3f,
    #            depth: mi.UInt32,
    #            δL: Optional[mi.Spectrum],
    #            state_in: Any,
    #            active: mi.Bool,
    #            project: bool = False,
    #            si_shade: Optional[mi.SurfaceInteraction3f] = None,
    #            **kwargs # Absorbs unused arguments
    # ) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float], Any]:
    #     """
    #     See ``PSIntegrator.sample()`` for a description of this interface and
    #     the role of the various parameters and return values.
    #     """

    #     # Rendering a primal image? (vs performing forward/reverse-mode AD)
    #     primal = mode == dr.ADMode.Primal

    #     # Should we use ``si_shade`` as the first interaction and ignore the ray?
    #     ignore_ray = si_shade is not None

    #     # Standard BSDF evaluation context for path tracing
    #     bsdf_ctx = mi.BSDFContext()

    #     # --------------------- Configure loop state ----------------------
    #     ray = mi.Ray3f(dr.detach(ray))                # Current ray
    #     depth = mi.UInt32(depth)                      # Depth of the current path segment
    #     L = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
    #     δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
    #     β = mi.Spectrum(1)                            # Path throughput weight
    #     η = mi.Float(1)                               # Index of refraction
    #     active = mi.Bool(active)                      # Active SIMD lanes
    #     depth_init = mi.UInt32(depth)                 # Initial depth

    #     # Variables caching information
    #     prev_ray        = mi.Ray3f(dr.detach(ray))
    #     prev_bsdf_pdf   = mi.Float(1.0)
    #     prev_bsdf_delta = mi.Bool(True)

    #     # Projective seed ray information
    #     cnt_seed = mi.UInt32(0)            # Number of valid seed rays encountered
    #     ray_seed = dr.zeros(mi.Ray3f)      # Seed ray to be projected
    #     active_seed = mi.Bool(active)      # Active SIMD lanes

    #     while dr.hint(active,
    #                   max_iterations=self.max_depth,
    #                   label="PRB Projective Fix (%s)" % mode.name):
    #         active_next = mi.Bool(active)

    #         # Compute a surface interaction that tracks derivatives arising
    #         # from differentiable shape parameters (position, normals, etc).
    #         # In primal mode, this is just an ordinary ray tracing operation.
    #         use_si_shade = ignore_ray & (depth == depth_init)
    #         with dr.resume_grad(when=not primal):
    #             prev_si = scene.ray_intersect(prev_ray,
    #                                      ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
    #                                      coherent=(depth == 0))
    #             si = scene.ray_intersect(ray,
    #                                      ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
    #                                      coherent=(depth == 0))
    #             # si.wi has a gradient as prev_si might move with pi
    #             # if dr.hint(not primal, mode='scalar'):
    #             si.wi = dr.select((depth == 0) | ~si.is_valid(), si.wi, si.to_local(dr.normalize(prev_si.p - si.p)))
    #         if dr.hint(ignore_ray, mode='scalar'):
    #             si[use_si_shade] = si_shade

    #         # Get the BSDF, potentially computes texture-space differentials
    #         bsdf = si.bsdf(ray)

    #         # ---------------------- Direct emission ----------------------

    #         # Hide the environment emitter if necessary
    #         if dr.hint(self.hide_emitters, mode='scalar'):
    #             active_next &= ~((depth == 0) & ~si.is_valid())

    #         # Compute MIS weight for emitter sample from previous bounce
    #         ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

    #         si_pdf = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            
            
    #         with dr.resume_grad(when=not primal):
    #             dist_squared = dr.squared_norm(si.p-prev_si.p)
    #             dp = dr.dot(si.to_world(si.wi), si.n)
    #             # For environment emitters, si.is_valid() will be false and
    #             # the local frame (dp_du, dp_dv) is invalid, but they should still contribute
    #             D = dr.select(active_next & si.is_valid(), dr.norm(dr.cross(si.dp_du, si.dp_dv)) * dp / dist_squared, 1.)
    #             LD = L * dr.replace_grad(1., D/dr.detach(D))
    #             LD = dr.select((depth == 0), 0, LD)           
            
    #         # MW: For completeness, include multiplication by D (but it cancels)
    #         mis = mis_weight(
    #             prev_bsdf_pdf*D,
    #             si_pdf*D
    #         )
    #         # The first samples are sampled differently
    #         # Ugo: neccessary? I think mis is 1 for first hit
    #         mis = dr.select((depth == 0), 1, mis)

    #         # remember beta contains geometry term/pdf == 1
    #         with dr.resume_grad(when=not primal):
    #             Le = β * mis * ds.emitter.eval(si, active_next)


    #         # ---------------------- Emitter sampling ----------------------

    #         # Should we continue tracing to reach one more vertex?
    #         active_next &= (depth + 1 < self.max_depth) & si.is_valid()
    #         # Is emitter sampling even possible on the current vertex?
    #         active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

    #         # If so, randomly sample an emitter without derivative tracking.
    #         ds_em, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
    #         active_em &= (ds_em.pdf != 0.0)

    #         with dr.resume_grad(when=not primal):
    #             # We need to recompute the sample with follow shape so it is a detached uv sample
    #             si_em = scene.ray_intersect(dr.detach(si.spawn_ray(ds_em.d)), 
    #                                         ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
    #                                         coherent=mi.Bool(False),
    #                                         active=active_em)

    #             # calculate the bsdf weight (for path througput) and pdf (for mis weighting)
    #             # MW: for environment emitters, we have si_em.p != ds_em.p (because si_em.p = 0)
    #             # TODO: How to handle this correctly?
    #             diff_em = dr.select(si_em.is_valid(), si_em.p - si.p, ds_em.p - si.p)
    #             ds_em.d = dr.normalize(diff_em)
    #             wo = si.to_local(ds_em.d)
    #             bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
                
    #             # ds_em.pdf includes the inv geometry term, 
    #             # and bsdf_pdf_em does not contain the geometry term.
    #             # -> We need to multiply both with the geometry term:
    #             dp_em = dr.dot(ds_em.d, si_em.n)
    #             dist_squared_em = dr.squared_norm(diff_em)
    #             # For environment emitters, si.is_valid() will be false and
    #             # the local frame (dp_du, dp_dv) is invalid, but they should still contribute
    #             D_em = dr.select(active_em & si_em.is_valid(), dr.norm(dr.cross(si_em.dp_du, si_em.dp_dv)) * -dp_em / dist_squared_em , 1.) 
                
    #             if dr.hint(not primal, mode='scalar'):
    #                 # update gradient of em_weight
    #                 em_val = scene.eval_emitter_direction(si, ds_em, active_em)
    #                 em_weight = dr.replace_grad(em_weight, dr.select((ds_em.pdf != 0), em_val / ds_em.pdf, 0)) * dr.replace_grad(1, D_em/dr.detach(D_em))


    #         # MW: For completeness, include multiplication by D (but it cancels)
    #         mis_em = dr.select(ds_em.delta, 1, mis_weight(ds_em.pdf*D_em, bsdf_pdf_em*D_em))

    #         with dr.resume_grad(when=not primal):
    #             Lr_dir = β * mis_em * bsdf_value_em * em_weight


    #         # ------------------ Detached BSDF sampling -------------------

    #         bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
    #                                                sampler.next_1d(),
    #                                                sampler.next_2d(),
    #                                                active_next)

    #         # ---- Update loop variables based on current interaction -----

    #         L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)

    #         wo_world = si.to_world(bsdf_sample.wo)

    #         ray_next = si.spawn_ray(wo_world) 
    #         η *= bsdf_sample.eta
    #         # Detached Sampling
    #         β *= bsdf_weight

    #         # ------------------------ Seed rays --------------------------

    #         # Note: even when the current vertex has a delta BSDF, which implies
    #         # that any perturbation of the direction will be invalid, we still
    #         # allow projection of such a ray segment. If this is not desired,
    #         # mask ``active_seed_cand &= ~prev_bsdf_delta``.

    #         if dr.hint(project, mode='scalar'):
    #                 # Given the detached emitter sample, *recompute* its
    #             # Is it possible to project a seed ray from the current vertex?
    #             active_seed_cand = mi.Mask(active_next)

    #             # If so, pick a seed ray (if applicable)
    #             if dr.hint(self.project_seed == "bsdf", mode='scalar'):
    #                 # We assume that `ray` intersects the scene
    #                 ray_seed_cand = ray
    #             elif dr.hint(self.project_seed == "emitter", mode='scalar'):
    #                 ray_seed_cand = si.spawn_ray_to(ds.p)
    #                 ray_seed_cand.maxt = dr.largest(mi.Float)
    #                 # Directions towards the interior have no contribution
    #                 # unless we hit a transmissive BSDF
    #                 active_seed_cand &= (dr.dot(si.n, ray_seed_cand.d) > 0) | \
    #                                      mi.has_flag(bsdf.flags(), mi.BSDFFlags.Transmission)
    #             elif dr.hint(self.project_seed == "both", mode='scalar'):
    #                 # By default we use the BSDF sample as the seed ray
    #                 ray_seed_cand = ray

    #                 # Flip a coin only when the emitter sample is valid
    #                 mask_replace = (dr.dot(si.n, ray_seed_cand.d) > 0) | \
    #                                 mi.has_flag(bsdf.flags(), mi.BSDFFlags.Transmission)
    #                 mask_replace &= sampler.next_1d(active_seed_cand) > 0.5
    #                 ray_seed_cand[mask_replace] = si.spawn_ray_to(ds.p)
    #                 ray_seed_cand.maxt = dr.largest(mi.Float)

    #             # Resovoir sampling: do we use this candidate seed ray?
    #             cnt_seed[active_seed_cand] += 1
    #             replace = active_seed_cand & \
    #                       (sampler.next_1d(active_seed_cand) * cnt_seed <= 1)

    #             # Update the seed ray if necessary
    #             ray_seed[replace] = ray_seed_cand
    #             active_seed |= replace

    #         # -------------------- Stopping criterion ---------------------

    #         # Don't run another iteration if the throughput has reached zero
    #         β_max = dr.max(β)
    #         active_next &= (β_max != 0)

    #         # Russian roulette stopping probability (must cancel out ior^2
    #         # to obtain unitless throughput, enforces a minimum probability)
    #         rr_prob = dr.minimum(β_max * η**2, .95)

    #         # Apply only further along the path since, this introduces variance
    #         rr_active = depth >= self.rr_depth
    #         β[rr_active] *= dr.rcp(rr_prob)
    #         rr_continue = sampler.next_1d() < rr_prob
    #         active_next &= ~rr_active | rr_continue

    #         # ------------------ Differential phase only ------------------

    #         if dr.hint(not primal, mode='scalar'):
    #             with dr.resume_grad():
    #                 # 'L' stores the indirectly reflected radiance at the
    #                 # current vertex but does not track parameter derivatives.
    #                 # The following addresses this by canceling the detached
    #                 # BSDF value and replacing it with an equivalent term that
    #                 # has derivative tracking enabled. (nit picking: the
    #                 # direct/indirect terminology isn't 100% accurate here,
    #                 # since there may be a direct component that is weighted
    #                 # via multiple importance sampling)
    #                 si_next = scene.ray_intersect(ray_next,
    #                                               ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
    #                                               coherent=mi.Bool(False))
                    
    #                 # Recompute 'wo' to propagate derivatives to cosine term
    #                 diff_next = si_next.p - si.p
    #                 dir_next = dr.normalize(diff_next)
    #                 wo = si.to_local(dir_next)

    #                 # Re-evaluate BSDF * cos(theta) differentiably
    #                 bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next & si_next.is_valid())

    #                 # Detached version of the above term and inverse
    #                 bsdf_val_det = bsdf_weight * bsdf_sample.pdf
    #                 inv_bsdf_val_det = dr.select(bsdf_val_det != 0,
    #                                              dr.rcp(bsdf_val_det), 0)

    #                 # Differentiable version of the reflected indirect
    #                 # radiance. Minor optional tweak: indicate that the primal
    #                 # value of the second term is always 1.
    #                 tmp = inv_bsdf_val_det * bsdf_val
    #                 tmp_replaced = dr.replace_grad(dr.ones(mi.Float, dr.width(tmp)), tmp) #FIXME
    #                 Lr_ind = L * tmp_replaced

    #                 # Differentiable Monte Carlo estimate of all contributions
    #                 Lo = Le + LD + Lr_dir + Lr_ind

    #                 attached_contrib = dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo)
    #                 if dr.hint(attached_contrib, mode='scalar'):
    #                     raise Exception(
    #                         "The contribution computed by the differential "
    #                         "rendering phase is not attached to the AD graph! "
    #                         "Raising an exception since this is usually "
    #                         "indicative of a bug (for example, you may have "
    #                         "forgotten to call dr.enable_grad(..) on one of "
    #                         "the scene parameters, or you may be trying to "
    #                         "optimize a parameter that does not generate "
    #                         "derivatives in detached PRB.)")

    #                 # Propagate derivatives from/to 'Lo' based on 'mode'
    #                 if dr.hint(mode == dr.ADMode.Backward, mode='scalar'):
    #                     dr.backward_from(δL * Lo)
    #                 else:
    #                     δL += dr.forward_to(Lo)

    #         # ------------------ Prepare next iteration ------------------
    #         # Information about the current vertex needed by the next iteration
    #         prev_bsdf_pdf = bsdf_sample.pdf
    #         prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
            
    #         depth[si.is_valid()] += 1
    #         active = active_next
    #         prev_ray = ray
    #         ray = ray_next

    #     return (
    #         L if primal else δL, # Radiance/differential radiance
    #         depth != 0,          # Ray validity flag for alpha blending
    #         [],                  # Empty tuple of AOVs.
    #                              # Seed rays, or the state for the differential phase
    #         [dr.detach(ray_seed), active_seed] if project else L
    #     )

mi.register_integrator("prb_projective_temp", lambda props: PathProjectiveFixIntegrator(props))

del PSIntegrator
