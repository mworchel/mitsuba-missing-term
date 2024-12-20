from __future__ import annotations as __annotations__

import drjit as dr
import mitsuba as mi

from mitsuba.ad.integrators.common import ADIntegrator, mis_weight
import gc

class ThreePointIntegrator(ADIntegrator):
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
        

        for it in range(self.max_depth):
            active_next = mi.Bool(active)

            si = scene.ray_intersect(dr.detach(ray),
                                    ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                    coherent=(depth == 0))
            si.wi = dr.select((depth == 0) | ~active_next, si.wi, si.to_local(dr.normalize(ray.o - si.p))) 

            if it == 0:
                first_si = si
            

            # Get the BSDF, potentially computes texture-space differentials
            # Ugo: This ray needs to be constructed from prev_si/attached
            bsdf = si.bsdf(ray)

            # ---------------------- Direct emission ----------------------

            # Hide the environment emitter if necessary
            if self.hide_emitters:
                active_next &= ~((depth == 0) & ~si.is_valid())

            # Compute MIS weight for emitter sample from previous bounce
            ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

            si_pdf = scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta)
            dr.disable_grad(si_pdf)
            
            # Adopt the pdf as ds.pdf includes the inv geometry term, 
            # and prev_bsdf_pdf does not contain the geometry term.
            # -> We need to multiply both with the geometry term:
            dist_squared = dr.squared_norm(si.p-ray.o)
            dp = dr.dot(ds.d, ds.n)
            G = dr.select(active_next, dr.norm(dr.cross(si.dp_du, si.dp_dv)) * -dp / dist_squared , 1.)

            mis = mis_weight(
                prev_bsdf_pdf*G,
                si_pdf*G
            )
            # The first samples are sampled from screen space and not solid angles
            # -> We need to adopt the mis weight
            mis = dr.select((depth == 0), 1, mis)
            # G = dr.select((depth == 0), 1, G)
            if it != 0:
                β *= dr.replace_grad(1, G/dr.detach(G))

            #Le = β * si.emitter(scene).eval(si, active_next)
            Le = β * dr.detach(mis) * si.emitter(scene).eval(si, active_next)
            L += Le    

            # ---------------------- Attached Emitter sampling ----------------------

            # Should we continue tracing to reach one more vertex?
            active_next &= (depth + 1 < self.max_depth) & si.is_valid()

            # Is emitter sampling even possible on the current vertex?
            active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

            # If so, randomly sample an emitter with derivative tracking.
            ds_em, em_weight = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
            #em_weight /= G_em
            em_weight *= dr.replace_grad(1, ds_em.pdf/dr.detach(ds_em.pdf))

            active_em &= (ds_em.pdf != 0.0)
            
            # We need to recompute the sample just for si_em.dp_du, si_em.dp_dv
            # si_em should be equivalent to ds_em
            si_em = scene.ray_intersect(dr.detach(si.spawn_ray(ds_em.d)), 
                                        ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                        coherent=mi.Bool(False),
                                        active=active_em)

            diff_em = ds_em.p - si.p
            ds_em_dir = dr.normalize(diff_em)
            wo = si.to_local(dr.normalize(diff_em))
            bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)

            # This mis weight is wrong:
            # mis_em = dr.select(ds_em.delta, 1, mis_weight(ds_em.pdf, bsdf_pdf_em))

            # ds_em.pdf includes the inv geometry term, 
            # and bsdf_pdf_em does not contain the geometry term.
            # -> We need to multiply both with the geometry term:
            # dp_em = dr.dot(ds_em.d, ds_em.n)
            dp_em = dr.dot(ds_em_dir, ds_em.n)
            dist_squared_em = dr.squared_norm(diff_em)
            G_em = dr.select(active_em, dr.norm(dr.cross(si_em.dp_du, si_em.dp_dv)) * -dp_em / dist_squared_em , 0.)
            mis_em = dr.select(ds_em.delta, 1, mis_weight(ds_em.pdf*G_em, bsdf_pdf_em*G_em))
            # Detached Sampling
            em_weight *= dr.replace_grad(1, G_em/dr.detach(G_em))

            # As we sample (uv) points and not solid angles we need to add the geometry term
            # mis_em *
            Lr_dir = β * dr.detach(mis_em) * bsdf_value_em * em_weight # * G_em # * mis_em
            L += Lr_dir

            # ------------------ Attached BSDF sampling -------------------

            with dr.suspend_grad():
                bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
                                                    sampler.next_1d(),
                                                    sampler.next_2d(),
                                                    active_next)
            
            # The sampled bsdf direction and the pdf must be detached
            # Recompute `bsdf_weight = bsdf_val / bsdf_sample.pdf` with only `bsdf_val` attached
            # dr.disable_grad(bsdf_sample.wo, bsdf_sample.pdf)
            # bsdf_val    = bsdf.eval(bsdf_ctx, si, bsdf_sample.wo, active_next)
            # bsdf_weight = dr.replace_grad(bsdf_weight, dr.select(dr.neq(bsdf_sample.pdf, 0), bsdf_val / bsdf_sample.pdf, 0))

            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

            si_next = scene.ray_intersect(dr.detach(ray),
                                            ray_flags=mi.RayFlags.All | mi.RayFlags.FollowShape,
                                            coherent=mi.Bool(False))
            
            # Recompute 'wo' to propagate derivatives to cosine term
            diff_next = si_next.p - si.p
            dir_next = dr.normalize(diff_next)
            wo = si.to_local(dir_next)
            bsdf_val    = bsdf.eval(bsdf_ctx, si, wo, active_next)
            bsdf_weight = dr.replace_grad(bsdf_weight, dr.select((bsdf_sample.pdf != 0), bsdf_val / dr.detach(bsdf_sample.pdf), 0))
            # ---- Update loop variables based on current interaction -----

            η *= bsdf_sample.eta
            # Detached Sampling
            β *= bsdf_weight

            # Information about the current vertex needed by the next iteration
            prev_si = si
            prev_bsdf_pdf = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)
            # si = si_next

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
            
            depth[si.is_valid()] += 1
            active = active_next
    
        return (
            L,                   # Radiance/differential radiance
            (depth != 0),    # Ray validity flag for alpha blending
            [],                  # Empty typle of AOVs
            first_si             # Necessary for screen position
        )

mi.register_integrator("ad_threepoint", lambda props: ThreePointIntegrator(props))
