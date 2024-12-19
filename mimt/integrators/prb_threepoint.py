from __future__ import annotations as __annotations__

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import RBIntegrator, ADIntegrator, mis_weight
import gc


class PRBThreePointIntegrator(RBIntegrator):

    # def render_forward(self: mi.SamplingIntegrator,
    #                    scene: mi.Scene,
    #                    params: Any,
    #                    sensor: Union[int, mi.Sensor] = 0,
    #                    seed: mi.UInt32 = 0,
    #                    spp: int = 0) -> mi.TensorXf:
    #     """
    #     Evaluates the forward-mode derivative of the rendering step.

    #     Forward-mode differentiation propagates gradients from scene parameters
    #     through the simulation, producing a *gradient image* (i.e., the derivative
    #     of the rendered image with respect to those scene parameters). The gradient
    #     image is very helpful for debugging, for example to inspect the gradient
    #     variance or visualize the region of influence of a scene parameter. It is
    #     not particularly useful for simultaneous optimization of many parameters,
    #     since multiple differentiation passes are needed to obtain separate
    #     derivatives for each scene parameter. See ``Integrator.render_backward()``
    #     for an efficient way of obtaining all parameter derivatives at once, or
    #     simply use the ``mi.render()`` abstraction that hides both
    #     ``Integrator.render_forward()`` and ``Integrator.render_backward()`` behind
    #     a unified interface.

    #     Before calling this function, you must first enable gradient tracking and
    #     furthermore associate concrete input gradients with one or more scene
    #     parameters, or the function will just return a zero-valued gradient image.
    #     This is typically done by invoking ``dr.enable_grad()`` and
    #     ``dr.set_grad()`` on elements of the ``SceneParameters`` data structure
    #     that can be obtained obtained via a call to
    #     ``mi.traverse()``.

    #     Parameter ``scene`` (``mi.Scene``):
    #         The scene to be rendered differentially.

    #     Parameter ``params``:
    #        An arbitrary container of scene parameters that should receive
    #        gradients. Typically this will be an instance of type
    #        ``mi.SceneParameters`` obtained via ``mi.traverse()``. However, it
    #        could also be a Python list/dict/object tree (DrJit will traverse it
    #        to find all parameters). Gradient tracking must be explicitly enabled
    #        for each of these parameters using ``dr.enable_grad(params['parameter_name'])``
    #        (i.e. ``render_forward()`` will not do this for you). Furthermore,
    #        ``dr.set_grad(...)`` must be used to associate specific gradient values
    #        with each parameter.

    #     Parameter ``sensor`` (``int``, ``mi.Sensor``):
    #         Specify a sensor or a (sensor index) to render the scene from a
    #         different viewpoint. By default, the first sensor within the scene
    #         description (index 0) will take precedence.

    #     Parameter ``seed` (``int``)
    #         This parameter controls the initialization of the random number
    #         generator. It is crucial that you specify different seeds (e.g., an
    #         increasing sequence) if subsequent calls should produce statistically
    #         independent images (e.g. to de-correlate gradient-based optimization
    #         steps).

    #     Parameter ``spp`` (``int``):
    #         Optional parameter to override the number of samples per pixel for the
    #         differential rendering step. The value provided within the original
    #         scene specification takes precedence if ``spp=0``.
    #     """

    #     if isinstance(sensor, int):
    #         sensor = scene.sensors()[sensor]

    #     film = sensor.film()
    #     first_hit = sensor.film()

    #     # Disable derivatives in all of the following
    #     with dr.suspend_grad():
    #         # Prepare the film and sample generator for rendering
    #         sampler, spp = self.prepare(sensor, seed, spp, self.aov_names())

    #         # Generate a set of rays starting at the sensor, keep track of
    #         # derivatives wrt. sample positions ('pos') if there are any
    #         ray, weight, pos = self.sample_rays(scene, sensor, sampler)

    #         # Launch the Monte Carlo sampling process in primal mode (1)
    #         L, valid, aovs, state_out = self.sample(
    #             mode=dr.ADMode.Primal,
    #             scene=scene,
    #             sampler=sampler.clone(),
    #             ray=ray,
    #             depth=mi.UInt32(0),
    #             δL=None,
    #             state_in=None,
    #             active=mi.Bool(True)
    #         )
                
    #         # Launch the Monte Carlo sampling process in forward mode (2)
    #         δL, valid_2, δaovs, state_out_2 = self.sample(
    #             mode=dr.ADMode.Forward,
    #             scene=scene,
    #             sampler=sampler,
    #             ray=ray,
    #             depth=mi.UInt32(0),
    #             δL=None,
    #             δaovs=None,
    #             state_in=state_out,
    #             active=mi.Bool(True)
    #         )

    #         # Prepare an ImageBlock as specified by the film
    #         block = film.create_block()

    #         # Only use the coalescing feature when rendering enough samples
    #         block.set_coalesce(block.coalesce() and spp >= 4)

    #         # Accumulate into the image block
    #         ADIntegrator._splat_to_block(
    #             block, film, pos,
    #             value=δL * weight,
    #             weight=1.0,
    #             alpha=dr.select(valid_2, mi.Float(1), mi.Float(0)),
    #             aovs=[δaov * weight for δaov in δaovs],
    #             wavelengths=ray.wavelengths
    #         )
            
    #         # Perform the weight division and return an image tensor
    #         film.put_block(block)
    #         result_grad = film.develop()
            
    #         # import matplotlib.pyplot as plt#
    #         # import numpy as np
    #         # plt.imshow(result_grad, cmap='coolwarm', vmin=-np.quantile(np.abs(result_grad), 0.89), vmax=np.quantile(np.abs(result_grad), 0.89))
    #         # print(dr.max(result_grad))

    #         # Add differentials from first intersection
    #         with dr.resume_grad():


    #             # Prepare an ImageBlock as specified by the film
    #             first_hit_block = first_hit.create_block()

    #             # Only use the coalescing feature when rendering enough samples
    #             first_hit_block.set_coalesce(first_hit_block.coalesce() and spp >= 4)

    #             si = scene.ray_intersect(ray,
    #                                      ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
    #                                      coherent=mi.Bool(True))
                
    #             pos = dr.select(valid, sensor.sample_direction(si, [0, 0], active=valid)[0].uv, pos)
    #             diff = si.p-ray.o
    #             dist_squared = dr.squared_norm(diff)
    #             dp = dr.dot(dr.normalize(diff), si.n)
    #             G = dr.select(valid, dr.norm(dr.cross(si.dp_du, si.dp_dv)) * -dp / dist_squared , 1.)

    #             # Accumulate into the image first_hit_block
    #             ADIntegrator._splat_to_block(
    #                 first_hit_block, first_hit, pos,
    #                 value=L * weight * dr.replace_grad(1, G/dr.detach(G)),
    #                 weight=dr.replace_grad(1, G/dr.detach(G)),
    #                 alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
    #                 aovs=aovs,
    #                 wavelengths=ray.wavelengths
    #             )

    #             # Perform the weight division and return an image tensor
    #             first_hit.put_block(first_hit_block)
    #             result_img = first_hit.develop()

    #             # Propagate the gradients to the image tensor
    #             dr.forward_to(result_img)
    #             grad_first = dr.grad(result_img)

    #         # Explicitly delete any remaining unused variables
    #         del sampler, ray, weight, pos, L, valid, aovs, δL, δaovs, \
    #             valid_2, params, state_out, state_out_2, block

    #         result_grad += grad_first # ,flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)


    #     return result_grad


    def render_forward(self: mi.SamplingIntegrator,
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
            L, valid, aovs, state_out = self.sample(
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
            δL, valid_2, δaovs, state_out_2 = self.sample(
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

            # Prepare an ImageBlock as specified by the film
            block2 = film.create_block()

            # Only use the coalescing feature when rendering enough samples
            block2.set_coalesce(block2.coalesce() and spp >= 4)

            # Accumulate into the image block
            ADIntegrator._splat_to_block(
                block2, film, pos,
                value=δL * weight,
                weight=1.0,
                alpha=dr.select(valid_2, mi.Float(1), mi.Float(0)),
                aovs=[δaov * weight for δaov in δaovs],
                wavelengths=ray.wavelengths
            )
            
            # Perform the weight division and return an image tensor
            film.put_block(block2)
            result_grad = film.develop()
            
            film.clear()

            with dr.resume_grad():
                
                block = film.create_block()
                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)
                si = scene.ray_intersect(ray,
                                         ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                         coherent=mi.Bool(True))
                pos = dr.select(valid, sensor.sample_direction(si, [0, 0], active=valid)[0].uv, pos)
                dist_squared = dr.squared_norm(si.p-ray.o)
                dp = dr.dot(ray.d, si.n)
                G = dr.select(valid, dr.norm(dr.cross(si.dp_du, si.dp_dv)) * -dp / dist_squared , 1.)
                # Accumulate into the image block
                ADIntegrator._splat_to_block(
                    block, film, pos,
                    value=L * weight * dr.replace_grad(1, G/dr.detach(G)),
                    weight=dr.replace_grad(1, G/dr.detach(G)),
                    alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                    aovs=aovs,
                    wavelengths=ray.wavelengths
                )
                # Perform the weight division
                film.put_block(block)
                
                result_img = film.develop()

                # Propagate the gradients to the image tensor
                dr.forward_to(result_img, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                first_hit = dr.grad(result_img)

        return result_grad + first_hit
    
    def render_backward(self: mi.SamplingIntegrator,
                        scene: mi.Scene,
                        params: Any,
                        grad_in: mi.TensorXf,
                        sensor: Union[int, mi.Sensor] = 0,
                        seed: mi.UInt32 = 0,
                        spp: int = 0) -> None:
        """
        Evaluates the reverse-mode derivative of the rendering step.

        Reverse-mode differentiation transforms image-space gradients into scene
        parameter gradients, enabling simultaneous optimization of scenes with
        millions of free parameters. The function is invoked with an input
        *gradient image* (``grad_in``) and transforms and accumulates these into
        the gradient arrays of scene parameters that previously had gradient
        tracking enabled.

        Before calling this function, you must first enable gradient tracking for
        one or more scene parameters, or the function will not do anything. This is
        typically done by invoking ``dr.enable_grad()`` on elements of the
        ``SceneParameters`` data structure that can be obtained obtained via a call
        to ``mi.traverse()``. Use ``dr.grad()`` to query the
        resulting gradients of these parameters once ``render_backward()`` returns.

        Parameter ``scene`` (``mi.Scene``):
            The scene to be rendered differentially.

        Parameter ``params``:
           An arbitrary container of scene parameters that should receive
           gradients. Typically this will be an instance of type
           ``mi.SceneParameters`` obtained via ``mi.traverse()``. However, it
           could also be a Python list/dict/object tree (DrJit will traverse it
           to find all parameters). Gradient tracking must be explicitly enabled
           for each of these parameters using ``dr.enable_grad(params['parameter_name'])``
           (i.e. ``render_backward()`` will not do this for you).

        Parameter ``grad_in`` (``mi.TensorXf``):
            Gradient image that should be back-propagated.

        Parameter ``sensor`` (``int``, ``mi.Sensor``):
            Specify a sensor or a (sensor index) to render the scene from a
            different viewpoint. By default, the first sensor within the scene
            description (index 0) will take precedence.

        Parameter ``seed` (``int``)
            This parameter controls the initialization of the random number
            generator. It is crucial that you specify different seeds (e.g., an
            increasing sequence) if subsequent calls should produce statistically
            independent images (e.g. to de-correlate gradient-based optimization
            steps).

        Parameter ``spp`` (``int``):
            Optional parameter to override the number of samples per pixel for the
            differential rendering step. The value provided within the original
            scene specification takes precedence if ``spp=0``.
        """

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()
        first_hit = sensor.film()

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
            L, valid, aovs, state_out = self.sample(
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
            L_2, valid_2, aovs_2, state_out_2 = self.sample(
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

                dist_squared = dr.squared_norm(si.p-ray.o)
                dp = dr.dot(ray.d, si.n)
                G = dr.select(valid, dr.norm(dr.cross(si.dp_du, si.dp_dv)) * -dp / dist_squared , 1.)

            # Prepare an ImageBlock as specified by the film
            block = first_hit.create_block()

            # Only use the coalescing feature when rendering enough samples
            block.set_coalesce(block.coalesce() and spp >= 4)

            # Accumulate into the image block
            ADIntegrator._splat_to_block(
                block, first_hit, pos,
                value=L * weight * dr.replace_grad(1, G/dr.detach(G)),
                weight=dr.replace_grad(1, G/dr.detach(G)),
                alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                aovs=aovs,
                wavelengths=ray.wavelengths
            )

            # Perform the weight division and return an image tensor
            first_hit.put_block(block)
            image = film.develop()

            dr.set_grad(image, grad_in)
            dr.enqueue(dr.ADMode.Backward, image)
            dr.traverse(dr.ADMode.Backward)
            
            # We don't need any of the outputs here
            del L_2, valid_2, aovs_2, state_out, state_out_2, \
                δL, δaovs, ray, weight, pos, sampler


            # Run kernel representing side effects of the above
            dr.eval()

    @dr.syntax
    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               δL: Optional[mi.Spectrum],
               state_in: Optional[mi.Spectrum],
               active: mi.Bool,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float], mi.Spectrum]:
        
        # Rendering a primal image? (vs performing forward/reverse-mode AD)
        primal = mode == dr.ADMode.Primal

        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0 if primal else state_in)    # Radiance accumulator
        δL = mi.Spectrum(δL if δL is not None else 0) # Differential/adjoint radiance
        β = mi.Spectrum(1)                            # Path throughput weight (includes geometry terms)
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_ray        = mi.Ray3f(dr.detach(ray))
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        prev_G          = mi.Float(1.0)
        
        while dr.hint(active,
                      max_iterations=self.max_depth,
                      label="Path Replay Backpropagation (%s)" % mode.name):
            active_next = mi.Bool(active)

            with dr.resume_grad(when=not primal):
                prev_si = scene.ray_intersect(prev_ray,
                                         ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                         coherent=(depth == 0))
                si = scene.ray_intersect(ray,
                                         ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                         coherent=(depth == 0))
                # si.wi has a gradient as prev_si might move with pi
                # if dr.hint(not primal, mode='scalar'):
                si.wi = dr.select((depth == 0), si.wi, dr.normalize(prev_si.p - si.p))
                    # ray.o = prev_si.p

            # Get the BSDF, potentially computes texture-space differentials
            # Ugo: Do we need this in three point form? I thought bsdf is detached
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            # Hide the environment emitter if necessary
            if dr.hint(self.hide_emitters, mode='scalar'):
                active_next &= ~((depth == 0) & ~si.is_valid())

            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            
            si_pdf = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            
            
            with dr.resume_grad(when=not primal):
                dist_squared = dr.squared_norm(si.p-prev_si.p)
                dp = dr.dot(si.wi, si.n)
                G = dr.select(active_next, dr.norm(dr.cross(si.dp_du, si.dp_dv)) * dp / dist_squared , 1.)
            
            mis = mis_weight(
                prev_bsdf_pdf*G,
                si_pdf*G
            )
            # The first samples are sampled differently
            # Ugo: neccessary? I think mis is 1 for first hit
            mis = dr.select((depth == 0), 1, mis)

            # remember beta contains geometry term/pdf == 1
            with dr.resume_grad(when=not primal):
                Le = β * mis * si.emitter(scene).eval(si, active_next)

            # ---------------------- Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next &= (depth + 1 < self.max_depth) & si.is_valid()
            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter without derivative tracking.
            ds_em, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
            active_em &= (ds_em.pdf != 0.0)
            
            with dr.resume_grad(when=not primal):
                # We need to recompute the sample with follow shape so it is a detached uv sample
                si_em = scene.ray_intersect(dr.detach(si.spawn_ray(ds_em.d)), 
                                            ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                            coherent=mi.Bool(False),
                                            active=active_em)

                # calculate the bsdf weight (for path througput) and pdf (for mis weighting)
                diff_em = si_em.p - si.p
                ds_em.d = dr.normalize(diff_em)
                wo = si.to_local(dr.normalize(ds_em.d))
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
                
                # ds_em.pdf includes the inv geometry term, 
                # and bsdf_pdf_em does not contain the geometry term.
                # -> We need to multiply both with the geometry term:
                dp_em = dr.dot(ds_em.d, si_em.n)
                dist_squared_em = dr.squared_norm(diff_em)
                G_em = dr.select(active_em, dr.norm(dr.cross(si_em.dp_du, si_em.dp_dv)) * -dp_em / dist_squared_em , 0.) 
                
                if dr.hint(not primal, mode='scalar'):
                    # update gradient of em_weight
                    em_val = scene.eval_emitter_direction(si, ds_em, active_em)
                    em_weight = dr.replace_grad(em_weight, dr.select((ds_em.pdf != 0), em_val / ds_em.pdf, 0)) * dr.replace_grad(1, G_em/dr.detach(G_em))


            mis_em = dr.select(ds_em.delta, 1, mis_weight(ds_em.pdf*G_em, bsdf_pdf_em*G_em))

            with dr.resume_grad(when=not primal):
                Lr_dir = β * mis_em * bsdf_value_em * em_weight

            # ------------------ BSDF sampling -------------------

            bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                sampler.next_1d(),
                                                sampler.next_2d(),
                                                active_next)

            # ---- Update loop variables based on current interaction -----

            L = (L + Le + Lr_dir) if primal else (L - Le - Lr_dir)

            wo_world = si.to_world(bsdf_sample.wo)

            ray_next = si.spawn_ray(wo_world) 
            η *= bsdf_sample.eta
            # Detached Sampling
            β *= bsdf_weight


            # -------------------- Stopping criterion ---------------------

            # Don't run another iteration if the throughput has reached zero
            β_max = dr.max(β)
            active_next &= (β_max != 0)

            # Russian roulette stopping probability (must cancel out ior^2
            # to obtain unitless throughput, enforces a minimum probability)
            rr_prob = dr.minimum(β_max * η**2, .95)

            # Apply only further along the path since, this introduces variance
            rr_active = depth >= self.rr_depth
            β[rr_active] *= dr.rcp(rr_prob)
            rr_continue = sampler.next_1d() < rr_prob
            active_next &= ~rr_active | rr_continue

            # ------------------ Differential phase only ------------------

            if dr.hint(not primal, mode='scalar'):
                with dr.resume_grad():
                    # 'L' stores the indirectly reflected radiance at the
                    # current vertex but does not track parameter derivatives.
                    # The following addresses this by canceling the detached
                    # BSDF value and replacing it with an equivalent term that
                    # has derivative tracking enabled. (nit picking: the
                    # direct/indirect terminology isn't 100% accurate here,
                    # since there may be a direct component that is weighted
                    # via multiple importance sampling)
                    si_next = scene.ray_intersect(ray_next,
                                                  ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                                  coherent=mi.Bool(False))
                    
                    # Recompute 'wo' to propagate derivatives to cosine term
                    diff_next = si_next.p - si.p
                    dir_next = dr.normalize(diff_next)
                    wo = si.to_local(dir_next)

                    # Re-evaluate BSDF * cos(theta) differentiably
                    bsdf_val = bsdf.eval(bsdf_ctx, si, wo, active_next & si_next.is_valid())

                    # Detached version of the above term and inverse
                    bsdf_val_det = bsdf_weight * bsdf_sample.pdf
                    inv_bsdf_val_det = dr.select(bsdf_val_det != 0,
                                                 dr.rcp(bsdf_val_det), 0)

                    # Differentiable version of the reflected indirect
                    # radiance. Minor optional tweak: indicate that the primal
                    # value of the second term is always 1.
                    tmp = inv_bsdf_val_det * bsdf_val
                    tmp_replaced = dr.replace_grad(dr.ones(mi.Float, dr.width(tmp)), tmp) #FIXME
                    Lr_ind = L * tmp_replaced

                    L_G = L * dr.replace_grad(1, G/dr.detach(G))
                    L_G = dr.select((depth == 0), 0, L_G)

                    # Differentiable Monte Carlo estimate of all contributions
                    Lo = Le + L_G + Lr_dir + Lr_ind

                    attached_contrib = dr.flag(dr.JitFlag.VCallRecord) and not dr.grad_enabled(Lo)
                    if dr.hint(attached_contrib, mode='scalar'):
                        raise Exception(
                            "The contribution computed by the differential "
                            "rendering phase is not attached to the AD graph! "
                            "Raising an exception since this is usually "
                            "indicative of a bug (for example, you may have "
                            "forgotten to call dr.enable_grad(..) on one of "
                            "the scene parameters, or you may be trying to "
                            "optimize a parameter that does not generate "
                            "derivatives in detached PRB.)")

                    # Propagate derivatives from/to 'Lo' based on 'mode'
                    if dr.hint(mode == dr.ADMode.Backward, mode='scalar'):
                        dr.backward_from(δL * Lo)
                    else:
                        δL += dr.forward_to(Lo)

            # ------------------ Prepare next iteration ------------------
            # Information about the current vertex needed by the next iteration
            prev_si = si
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
            
            # G term is needed for mis in next hit
            prev_G = G

            depth[si.is_valid()] += 1
            active = active_next
            ray = ray_next
    
        return (
            L if primal else δL, # Radiance/differential radiance
            (depth != 0),        # Ray validity flag for alpha blending
            [],                  # Empty typle of AOVs
            L                    # State for the differential phase
        )

mi.register_integrator("prb_threepoint", lambda props: PRBThreePointIntegrator(props))
