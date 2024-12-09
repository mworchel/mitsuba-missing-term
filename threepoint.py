from __future__ import annotations as __annotations__

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import ADIntegrator, mis_weight
import gc

class ThreePointIntegrator(ADIntegrator):

    def render(self: mi.SamplingIntegrator,
               scene: mi.Scene,
               sensor: Union[int, mi.Sensor] = 0,
               seed: mi.UInt32 = 0,
               spp: int = 0,
               develop: bool = True,
               evaluate: bool = True) -> mi.TensorXf:
        if not develop:
            raise Exception("develop=True must be specified when "
                            "invoking AD integrators")

        if isinstance(sensor, int):
            sensor = scene.sensors()[sensor]

        film = sensor.film()

        # Disable derivatives in all of the following
        with dr.suspend_grad():
            # Prepare the film and sample generator for rendering
            sampler, spp = self.prepare(
                sensor=sensor,
                seed=seed,
                spp=spp,
                aovs=self.aov_names()
            )

            # Generate a set of rays starting at the sensor
            ray, weight, pos = self.sample_rays(scene, sensor, sampler)

            # Launch the Monte Carlo sampling process in primal mode
            L, valid, aovs, _ = self.sample(
                mode=dr.ADMode.Primal,
                scene=scene,
                sampler=sampler,
                ray=ray,
                depth=mi.UInt32(0),
                δL=None,
                δaovs=None,
                state_in=None,
                active=mi.Bool(True)
            )

            # Prepare an ImageBlock as specified by the film
            block = film.create_block()

            # Only use the coalescing feature when rendering enough samples
            block.set_coalesce(block.coalesce() and spp >= 4)

            # Accumulate into the image block
            ADIntegrator._splat_to_block(
                block, film, pos,
                value=L * weight,
                weight=1.0,
                alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                aovs=aovs,
                wavelengths=ray.wavelengths
            )

            # Explicitly delete any remaining unused variables
            del sampler, ray, weight, pos, L, valid

            # Perform the weight division and return an image tensor
            film.put_block(block)

            return film.develop()

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

            with dr.resume_grad():
                L, valid, aovs, si = self.sample(
                    mode=dr.ADMode.Forward,
                    scene=scene,
                    sampler=sampler,
                    ray=ray,
                    active=mi.Bool(True)
                )
                
                block = film.create_block()
                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)

                pos = dr.select(valid, sensor.sample_direction(si, [0, 0], active=valid)[0].uv, pos)

                ADIntegrator._splat_to_block(
                    block, film, pos,
                    value=L * weight,
                    weight=1,
                    alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                    aovs=aovs,
                    wavelengths=ray.wavelengths
                )
                # Perform the weight division
                film.put_block(block)
                result_img = film.develop()

                # Propagate the gradients to the image tensor
                dr.forward_to(result_img)

        return dr.grad(result_img)

    def render_backward(self: mi.SamplingIntegrator,
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

            with dr.resume_grad():
                L, valid, aovs, si = self.sample(
                    mode=dr.ADMode.Backward,
                    scene=scene,
                    sampler=sampler,
                    ray=ray,
                    active=mi.Bool(True)
                )

                # Prepare an ImageBlock as specified by the film
                block = film.create_block()

                # Only use the coalescing feature when rendering enough samples
                block.set_coalesce(block.coalesce() and spp >= 4)

                pos = dr.select(valid, sensor.sample_direction(si, [0, 0], active=valid)[0].uv, pos)

                # Accumulate into the image block
                ADIntegrator._splat_to_block(
                    block, film, pos,
                    value=L * weight,
                    weight=1,
                    alpha=dr.select(valid, mi.Float(1), mi.Float(0)),
                    aovs=aovs,
                    wavelengths=ray.wavelengths
                )

                film.put_block(block)

                del valid

                # This step launches a kernel
                dr.schedule(block.tensor())
                image = film.develop()

                # Differentiate sample splatting and weight division steps to
                # retrieve the adjoint radiance
                dr.set_grad(image, grad_in)
                dr.enqueue(dr.ADMode.Backward, image)
                dr.traverse(dr.ADMode.Backward)

            # We don't need any of the outputs here
            del ray, weight, pos, block, sampler

            # Run kernel representing side effects of the above
            dr.eval()

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               active: mi.Bool,
               **kwargs # Absorbs unused arguments
    ) -> Tuple[mi.Spectrum, mi.Bool, List[mi.Float], mi.Spectrum]:


        # Standard BSDF evaluation context for path tracing
        bsdf_ctx = mi.BSDFContext()

        # --------------------- Configure loop state ----------------------

        # Copy input arguments to avoid mutating the caller's state
        ray = mi.Ray3f(dr.detach(ray))
        depth = mi.UInt32(0)                          # Depth of current vertex
        L = mi.Spectrum(0)                            # Radiance accumulator
        
        β = mi.Spectrum(1)                            # Path throughput weight
        η = mi.Float(1)                               # Index of refraction
        active = mi.Bool(active)                      # Active SIMD lanes

        # Variables caching information from the previous bounce
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)
        

        with dr.resume_grad():
            for iter in range(self.max_depth):
                active_next = mi.Bool(active)

                si = scene.ray_intersect(ray,
                                        ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                        coherent=dr.eq(depth, 0))

                if iter == 0:
                    first_si = si
                

                # Get the BSDF, potentially computes texture-space differentials
                bsdf = si.bsdf(ray)
    
                # ---------------------- Direct emission ----------------------
    
                # Hide the environment emitter if necessary
                if self.hide_emitters:
                    active_next &= ~(dr.eq(depth, 0) & ~si.is_valid())
    
                # Compute MIS weight for emitter sample from previous bounce
                ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

                si_pdf = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
                dr.disable_grad(si_pdf)
                
                # Adopt the pdf as ds.pdf includes the inv geometry term, 
                # and prev_bsdf_pdf does not contain the geometry term.
                # -> We need to multiply both with the geometry term:
                dist_squared = dr.squared_norm(ds.d);
                dp = dr.dot(ds.d, ds.n)
                G = dr.select(active_next, dr.norm(dr.cross(si.dp_du, si.dp_dv)) * -dp / dist_squared , 0.)

                mis = mis_weight(
                    prev_bsdf_pdf*G,
                    si_pdf*G
                )
                # The first samples are sampled from screen space and not solid angles
                # -> We need to adopt the mis weight
                mis = dr.select(dr.eq(depth,0 ), mis, mis_weight(prev_bsdf_pdf, si_pdf))
    
                Le = β * mis * si.emitter(scene).eval(si, active_next)#* G/G
                L += Le
                

                # ---------------------- Attached Emitter sampling ----------------------
    
                # Should we continue tracing to reach one more vertex?
                active_next &= (depth + 1 < self.max_depth) & si.is_valid()

                # Is emitter sampling even possible on the current vertex?
                active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

                # If so, randomly sample an emitter with derivative tracking.
                ds, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
                # This is ncessary for si_em.dp_du, si_em.dp_dv
                si_em = scene.ray_intersect(si.spawn_ray(ds.d), 
                                            ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                            coherent=mi.Bool(False),
                                            active=active_em)

                active_em &= dr.neq(ds.pdf, 0.0)

                # Ugo: ds.d = ds.p - si.p; and as ds is attached this is attached as well.
                wo = si.to_local(ds.d)
                bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)

                # This mis weight is wrong:             
                # mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))

                # ds.pdf includes the inv geometry term, 
                # and bsdf_pdf_em does not contain the geometry term.
                # -> We need to multiply both with the geometry term:
                dist_squared_em = dr.squared_norm(ds.d);
                dp_em = dr.dot(ds.d, ds.n)
                G_em = dr.select(active, dr.norm(dr.cross(si_em.dp_du, si_em.dp_dv)) * -dp_em / dist_squared_em , 0.)
                mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf*G_em, bsdf_pdf_em*G_em))

                # As we sample (uv) points and not solid angles we need to add the geometry term
                Lr_dir = β * mis_em * bsdf_value_em * em_weight#* G_em/G_em
                L += Lr_dir
                
                # ------------------ Attached BSDF sampling -------------------

                bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                    sampler.next_1d(),
                                                    sampler.next_2d(),
                                                    active_next)
                
                # The sampled bsdf direction and the pdf must be detached
                # Recompute `bsdf_weight = bsdf_val / bsdf_sample.pdf` with only `bsdf_val` attached
                # dr.disable_grad(bsdf_sample.wo, bsdf_sample.pdf)
                # bsdf_val    = bsdf.eval(bsdf_ctx, si, bsdf_sample.wo, active_next)
                # bsdf_weight = dr.replace_grad(bsdf_weight, dr.select(dr.neq(bsdf_sample.pdf, 0), bsdf_val / bsdf_sample.pdf, 0))
    
                # ---- Update loop variables based on current interaction -----
    
                wo_world = si.to_world(bsdf_sample.wo)
                # The direction in *world space* is detached
                dr.disable_grad(wo_world)

                ray = si.spawn_ray(wo_world) 
                η *= bsdf_sample.eta
                β *= bsdf_weight

                # Information about the current vertex needed by the next iteration

                prev_si = si
                prev_bsdf_pdf = bsdf_sample.pdf
                prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
    
                # -------------------- Stopping criterion ---------------------
    
                # Don't run another iteration if the throughput has reached zero
                β_max = dr.max(β)
                active_next &= dr.neq(β_max, 0)
    
                # Russian roulette stopping probability (must cancel out ior^2
                # to obtain unitless throughput, enforces a minimum probability)
                rr_prob = dr.minimum(β_max * η**2, .95)
    
                # Apply only further along the path since, this introduces variance
                rr_active = depth >= self.rr_depth
                β[rr_active] *= dr.rcp(rr_prob)
                rr_continue = sampler.next_1d() < rr_prob
                active_next &= ~rr_active | rr_continue
                
                depth[si.is_valid()] += 1
                active = active_next
    
        return (
            L,                   # Radiance/differential radiance
            dr.neq(depth, 0),    # Ray validity flag for alpha blending
            [],                  # Empty typle of AOVs
            first_si             # Necessary for screen position
        )

mi.register_integrator("ad_threepoint", lambda props: ThreePointIntegrator(props))
