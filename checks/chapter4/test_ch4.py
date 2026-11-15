"""Native transition, information, risk, optimization and recovery-contract checks."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import random
import unittest
import uuid
from collections import deque
import numpy as np
import torch
from chapter4_interval_robust.configuration import ROOT,load,reference_config,validate,evaluation_contract,comparison_contract
from chapter4_interval_robust.parameters import (NOMINAL,PublicSet,PrivilegedContext,joint,
                                                expand,normalize,validate_bounds)
from chapter4_interval_robust.dynamics import Actuator,DisturbanceProcess,PhysicalIntervention,predict_velocity
from chapter4_interval_robust.estimation import History,HistorySetEstimator,EnsembleIdentifier
from chapter4_interval_robust.learning import MultiAgentSAC,lower_tail,ContextSupport
from chapter4_interval_robust.runtime import PolicyInput,ReferenceRuntime,public_nominal,public_phase_codes,OBS_DIM
from chapter4_interval_robust.scenarios import make_scene
from chapter4_interval_robust.metrics import tail_mean,wilson
from chapter4_interval_robust.protection import PublicProtection
from chapter4_interval_robust.provenance import digest
from chapter4_interval_robust import checkpoint

LEDGER=dict(native_environment_initializations=0,native_physical_steps=0,
            nominal_pair_passed=False,models={},historical_checkpoints_loaded=False,
            formal_training_started=False)


def model_ledger(name,model):
    LEDGER['models'][name]=dict(complete_updates=model.updates,maximum_updates=4,
                              physical_steps=model.environment_steps,episodes=0)
    assert model.updates<=4


def replay_fixture(model,count=16):
    """Synthetic optimizer fixture, never reported as physical/performance evidence."""
    rng=np.random.default_rng(5981);c=model.config
    prior=PublicSet.prior(model.bounds)
    for i in range(count):
        obs=rng.normal(0,.1,(4,OBS_DIM)).astype(np.float32);obs[:,39:42]=0
        obs[:,43:]=prior.features(model.bounds)
        active=np.array([True,True,True,False])
        theta=np.tile(NOMINAL,(4,1)).astype(float)
        # Physically distinct contexts; shared fields stay one joint realization.
        theta[:,:2]=rng.uniform(model.bounds[:2,0],model.bounds[:2,1])
        theta[:,2:]=rng.uniform(model.bounds[2:,0],model.bounds[2:,1],(4,7));theta[:,7]=np.round(theta[:,7])
        context=joint(normalize(theta,model.bounds)).astype(np.float32)
        h=rng.normal(0,.1,(c['estimation']['history'],4,12)).astype(np.float32)
        row=dict(observations=obs,active=active,proposal=np.zeros((4,3),np.float32),reward=1.,
                 next_observations=obs.copy(),next_active=active,terminated=bool(i%2),truncated=not bool(i%2),
                 context=context,next_context=context.copy(),next_history=h,
                 next_history_mask=np.ones(h.shape[:2],bool),privileged_target=theta,set_before=prior.as_dict())
        model.remember(row)


class Contracts(unittest.TestCase):
    def test_parameter_types_and_joint_world(self):
        c=load(mechanism=True);b=validate_bounds(c['disturbance']['bounds'])
        bad=b.copy();bad[7,1]=1.5
        with self.assertRaises(ValueError):validate_bounds(bad)
        truth=np.tile(NOMINAL,(4,1));np.testing.assert_array_equal(expand(joint(truth)),truth)
        truth[2,0]+=.01
        with self.assertRaises(ValueError):PrivilegedContext(truth)
        prior=PublicSet.prior(b);samples=prior.joint_samples(np.random.default_rng(2),64)
        np.testing.assert_array_equal(samples[:,:,:2],np.repeat(samples[:,:1,:2],4,axis=1))
        self.assertTrue(np.all(samples[:,:,7]==np.floor(samples[:,:,7])))

    def test_delay_lag_reset_sequence(self):
        a=Actuator(2);x=np.tile(NOMINAL,(4,1));x[:,7]=2;x[:,6]=.5
        u=np.ones((4,3));zero=np.zeros((4,3))
        np.testing.assert_array_equal(a.apply(u,x,zero),zero)
        np.testing.assert_array_equal(a.apply(zero,x,zero),zero)
        np.testing.assert_allclose(a.apply(zero,x,zero),.5)
        np.testing.assert_allclose(a.apply(zero,x,zero),.25)
        a.reset();np.testing.assert_array_equal(a.filtered,zero)
        self.assertTrue(all(not q.any() for q in a.queue))

    def test_disturbance_protocols(self):
        c=load(mechanism=True);o=copy.deepcopy(c['disturbance']);o['protocol']='piecewise'
        p=DisturbanceProcess(o,19);first=p.at(0).local.copy()
        np.testing.assert_array_equal(p.at(1).local,first)
        self.assertFalse(np.array_equal(p.at(8).local,first))
        p.actuator.filtered.fill(.4);p.reset();self.assertFalse(p.actuator.filtered.any())
        o['protocol']='drift';p=DisturbanceProcess(o,19);a=p.at(0).local.copy();b=p.at(1).local
        self.assertTrue(np.all(np.abs(b-a)<=np.array(o['drift_rate'])+1e-12))
        np.testing.assert_array_equal(a[:,7],b[:,7])

    def test_heldout_world_does_not_change_public_prior(self):
        c=load(mechanism=True);old=np.asarray(c['disturbance']['bounds']).copy()
        c['domain']='heldout_response';c=validate(c)
        np.testing.assert_array_equal(c['disturbance']['bounds'],old)
        process=DisturbanceProcess(c['disturbance'],19);truth=process.at(0).local
        np.testing.assert_array_equal(truth[:,7],3)
        self.assertEqual(process.actuator.max_delay,3)
        self.assertTrue(np.all(PublicSet.prior(old).upper[:,7]==2))

    def test_evaluation_contract_prevents_train_scene_relabel(self):
        train=load(mechanism=True);evaluation=copy.deepcopy(train)
        evaluation['splits']['train'],evaluation['splits']['dev']=evaluation['splits']['dev'],evaluation['splits']['train']
        validate(evaluation) # Individually legal, but not legal for these weights.
        with self.assertRaises(ValueError):evaluation_contract(train,evaluation)
        for key,value in [('max_steps',32),('reference_config','configs/b3.json'),('seed',9001)]:
            changed=copy.deepcopy(train);changed[key]=value
            with self.assertRaises(ValueError):evaluation_contract(train,changed)
        legal=copy.deepcopy(train);legal['domain']='heldout_flow';legal['disturbance']['protocol']='piecewise'
        evaluation_contract(train,legal)
        changed=copy.deepcopy(train);changed['disturbance']['change_steps']=[3]
        with self.assertRaises(ValueError):evaluation_contract(train,changed)
        other_seed=copy.deepcopy(train);other_seed['seed']+=1
        self.assertEqual(comparison_contract(train),comparison_contract(other_seed))
        other_clock=copy.deepcopy(train);other_clock['max_steps']=32
        self.assertNotEqual(comparison_contract(train),comparison_contract(other_clock))

    def test_tail_orientation_gradient_and_support(self):
        q=torch.tensor([[4.,-3.,2.,-1.]],requires_grad=True)
        result=lower_tail(q,.5,2);self.assertEqual(float(result.detach()),-2.)
        result.backward();np.testing.assert_array_equal(q.grad.numpy(),[[0.,.5,0.,.5]])
        with self.assertRaises(ValueError):lower_tail(q,.01,2)
        self.assertEqual(float(lower_tail(q.detach(),1.,2)),.5)
        gate=ContextSupport(2,.1);query=np.zeros((1,2,30))
        gate.add(np.zeros(30));self.assertFalse(gate.supported(query).any())
        gate.add(np.full(30,.01));self.assertTrue(gate.supported(query).all())
        self.assertFalse(gate.supported(query+1).any())
        self.assertEqual(tail_mean([4,-3,2,-1],.5),-2)
        self.assertTrue(wilson(0,10)[1]>0)

    def test_public_information_boundary(self):
        c=load(mechanism=True);m=MultiAgentSAC(c)
        x=np.zeros((4,OBS_DIM),np.float32);x[:,43:]=PublicSet.prior(m.bounds).features(m.bounds)
        public=PolicyInput(x,np.ones(4,bool))
        expected=m.act(public,deterministic=True)
        unrelated_privileged=np.tile(NOMINAL,(4,1));unrelated_privileged[:,0]=.27
        np.testing.assert_array_equal(m.act(public,deterministic=True),expected)
        with self.assertRaises(TypeError):m.act({'observations':x,'truth':unrelated_privileged})
        self.assertNotIn('truth',inspect.signature(HistorySetEstimator.update).parameters)
        self.assertNotIn('truth',inspect.signature(EnsembleIdentifier.infer).parameters)
        self.assertEqual(set(PolicyInput.__dataclass_fields__),{'observations','active'})
        model_ledger('information_boundary',m)

    def test_phase_ablation_keeps_public_diagnostic_identity(self):
        class Task:
            target_found=True;executor_knows_target=True
        class PublicEnv:
            def get_task_state(self):return Task()
        obs=np.zeros((4,28));obs[3,21]=.5
        np.testing.assert_array_equal(public_phase_codes(PublicEnv(),obs),[1,1,1,3])
        # Diagnostic identity is derived before Actor-specific feature masking.
        c=load(method='phase_off',mechanism=True)
        self.assertFalse(c['method_options']['phase'])

    def test_estimator_invalid_samples_and_reset(self):
        c=load(mechanism=True)
        specs=[dict(a_xy_max=1.,a_z_max=.7,v_xy_max=2.,v_z_max=1.,drag_xy=.1,drag_z=.2,buoyancy_bias=0.)]*4
        estimator=HistorySetEstimator(c['disturbance']['bounds'],specs,{'dt':.2},c['estimation'],13)
        zero=np.zeros((4,3));prior=copy.deepcopy(estimator.estimate)
        result=estimator.update(zero,zero,zero,np.full((4,3),100.),np.zeros(4,bool))
        np.testing.assert_array_equal(result.lower,prior.lower);np.testing.assert_array_equal(result.upper,prior.upper)
        result=estimator.update(zero,zero,zero,np.full((4,3),100.),np.ones(4,bool))
        self.assertFalse(result.valid);self.assertIn('reset',result.reason)
        np.testing.assert_array_equal(result.lower,np.tile(estimator.bounds[:,0],(4,1)))

    def test_correct_delay_survives_history_consistency(self):
        c=load(mechanism=True);o=copy.deepcopy(c['estimation']);o.update(velocity_tolerance=.0001,hull_padding=0,minimum_particles=1)
        specs=[dict(a_xy_max=1.,a_z_max=.7,v_xy_max=3.,v_z_max=2.,drag_xy=.1,drag_z=.2,buoyancy_bias=0.)]*4
        e=HistorySetEstimator(c['disturbance']['bounds'],specs,{'dt':.2},o,13)
        e.particles[:]=NOMINAL;e.particles[:len(e.particles)//2,:,7]=1
        truth=np.tile(NOMINAL,(4,1));truth[:,7]=1
        actuator=Actuator(2);p=np.zeros((4,3));v=np.zeros((4,3))
        for t in range(3):
            command=np.full((4,3),.5 if t==0 else -.2)
            applied=actuator.apply(command,truth,np.zeros((4,3)))
            nv=predict_velocity(v,applied,truth,specs,.2,p)
            result=e.update(p,v,command,nv,np.ones(4,bool));p=p+nv*.2;v=nv
        self.assertTrue(result.valid)
        np.testing.assert_array_equal(result.lower[:,7],1)
        np.testing.assert_array_equal(result.upper[:,7],1)

    def test_ensemble_mean_spread_and_unsupported_role(self):
        c=load(mechanism=True);bounds=np.asarray(c['disturbance']['bounds'])
        model=EnsembleIdentifier(bounds,4,8,True)
        values=np.array([-.2,-.05,.05,.2])
        with torch.no_grad():
            for network,value in zip(model.models,values):
                for parameter in network.parameters():parameter.zero_()
                network[-1].bias.fill_(float(value))
        history=History(2);history.append(np.zeros((4,3)),np.zeros((4,3)),np.zeros((4,3)),
                                        np.zeros((4,3)),[True,True,True,False])
        prior=PublicSet.prior(bounds)
        untrained=model.infer(history,prior);self.assertFalse(untrained.valid)
        np.testing.assert_array_equal(untrained.lower,prior.lower)
        # The following hand-set ensemble is a mathematical inference fixture.
        model.updates=1
        estimate=model.infer(history,prior)
        sigma=float((.5*np.tanh(values)).std())
        span=bounds[:,1]-bounds[:,0]
        np.testing.assert_allclose(estimate.lower[0,:7],bounds[:7,0]+(.5-sigma)*span[:7],atol=1e-7)
        np.testing.assert_array_equal(estimate.lower[3],prior.lower[3]);np.testing.assert_array_equal(estimate.upper[3],prior.upper[3])
        frames,mask=history.arrays();loss=model.update_supervised(np.repeat(frames[None],4,axis=0),
                    np.repeat(mask[None],4,axis=0),np.tile(NOMINAL,(4,4,1)))
        self.assertTrue(np.isfinite(loss));self.assertEqual(model.updates,2)

    def test_public_probe_preserves_inactive_agents(self):
        guard=PublicProtection.__new__(PublicProtection)
        guard.specs=[dict(a_xy_max=1.,a_z_max=.7,v_xy_max=2.,v_z_max=1.,drag_xy=.1,drag_z=.2,buoyancy_bias=0.)]*4
        guard.contract={'dt':.2,'bounds':(20,20,8)};guard.options={'horizon':2}
        guard.commands=deque(maxlen=17);guard.rng=np.random.default_rng(4)
        p=np.full((4,3),4.);p[0]=[20,20,8];v=np.zeros((4,3));x=np.tile(NOMINAL,(4,4,1))
        class Geometry:
            def segment(self,a,b):return True
        self.assertTrue(guard._simulate(p,v,np.zeros((4,3)),x,Geometry(),np.array([False,True,True,True])))

    def test_sac_terminal_bootstrap_and_polyak(self):
        c=load(method='c_off',mechanism=True);c['sac']['initial_temperature']=1e-30
        m=MultiAgentSAC(c);replay_fixture(m,4)
        with torch.no_grad():
            for p in m.target.parameters():p.zero_()
            m.target.q1[-1].bias.fill_(2);m.target.q2[-1].bias.fill_(2)
        old_target=m.target.q1[-1].bias.detach().clone()
        metrics=m.update()
        self.assertAlmostEqual(metrics['target_mean'],1.95,places=5)
        expected=old_target*(1-c['sac']['tau'])+m.critic.q1[-1].bias*c['sac']['tau']
        torch.testing.assert_close(m.target.q1[-1].bias,expected)
        self.assertEqual(metrics['bootstrap_rows'],2);self.assertEqual(metrics['collector_truncation_rows'],2)
        model_ledger('terminal_bootstrap',m)

    def test_checkpoint_resume_identical_next_update(self):
        c=load(mechanism=True);c['risk']['support_radius']=2.;c['risk']['minimum_contexts']=2
        m=MultiAgentSAC(c);replay_fixture(m)
        first=m.update();self.assertGreater(first['risk_supported_rows'],0)
        folder=ROOT/'work/ch4'/('checkpoint_mechanism_'+uuid.uuid4().hex)
        folder.mkdir(parents=True)
        path=folder/'boundary.pt';checkpoint.save(path,m,c,1,[])
        payload=checkpoint.load(path,c)
        expected=m.update();expected_actor=copy.deepcopy(m.actor.state_dict());expected_q=copy.deepcopy(m.critic.state_dict())
        restored=MultiAgentSAC(c);checkpoint.restore(payload,restored)
        actual=restored.update()
        self.assertEqual(actual,expected)
        for name,value in restored.actor.state_dict().items():torch.testing.assert_close(value,expected_actor[name],rtol=0,atol=0)
        for name,value in restored.critic.state_dict().items():torch.testing.assert_close(value,expected_q[name],rtol=0,atol=0)
        model_ledger('checkpoint_original',m);model_ledger('checkpoint_restored',restored)
        bad=folder/'intentional_invalid.pt';bad.write_bytes(b'invalid local test checkpoint')
        bad.with_suffix('.json').write_text(path.with_suffix('.json').read_text(encoding='utf-8'),encoding='utf-8')
        with self.assertRaises(ValueError):checkpoint.load(bad,c)


class Native(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.c=load(mechanism=True);cls.scene=make_scene(cls.c,'dev',0)

    def core(self,clear=False):
        from chapter3_bser.learning.base_runtime import _make_base_env,seed_innovations
        seed_innovations(772)
        env=_make_base_env(reference_config(self.c));scene=copy.deepcopy(self.scene)
        if clear:
            scene.update(obstacles=[],use_obstacles=False,obstacle_count=0,obstacle_layout_id='none',scenario_id='ch4_unit_physics_clear')
        env.reset(scenario=scene)
        env.unwrapped.use_residual_prior=False;env.unwrapped._residual_scale.fill_(1.)
        LEDGER['native_environment_initializations']+=1
        return env

    def test_nominal_hook_exact_transitions_and_random_streams(self):
        traces=[]
        for hooked in (False,True):
            env=self.core();hook=None
            if hooked:
                o=copy.deepcopy(self.c['disturbance']);o['protocol']='nominal'
                hook=PhysicalIntervention(env.unwrapped,DisturbanceProcess(o,72))
            rows=[]
            try:
                for step in range(4):
                    obs,reward,done=env.step(torch.full((4,3),.1))
                    LEDGER['native_physical_steps']+=1
                    rows.append(dict(obs=np.asarray(obs).copy(),reward=np.asarray(reward).copy(),done=done,
                                     positions=np.asarray(env.get_agent_state().positions),velocities=np.asarray(env.get_agent_state().velocities),
                                     torch_rng=torch.get_rng_state().clone(),numpy_rng=copy.deepcopy(np.random.get_state()),python_rng=random.getstate()))
                    if all(done):break
                traces.append(rows)
            finally:
                if hook:hook.restore()
                env.close()
        self.assertEqual(len(traces[0]),len(traces[1]))
        for left,right in zip(*traces):
            for key in ('obs','reward','positions','velocities'):np.testing.assert_array_equal(left[key],right[key])
            self.assertEqual(left['done'],right['done']);self.assertTrue(torch.equal(left['torch_rng'],right['torch_rng']))
            self.assertEqual(left['python_rng'],right['python_rng'])
            for a,b in zip(left['numpy_rng'],right['numpy_rng']):np.testing.assert_array_equal(a,b)
        LEDGER['nominal_pair_passed']=True

    def test_original_D2_zero_reference_conversion(self):
        from chapter3_bser.learning.runtime import RepairedMissionRuntime
        class Zero:
            def step(self,observations,explore=True):return [torch.zeros((1,3)) for _ in range(4)]
        original=RepairedMissionRuntime(reference_config(self.c),self.scene,seed=713,episode_id=0)
        LEDGER['native_environment_initializations']+=1;left=[]
        try:
            for t in range(4):
                row=original.advance(Zero(),explore=False);LEDGER['native_physical_steps']+=1
                left.append((np.asarray(original.env.get_agent_state().positions),row['rewards'],row['dones']))
                if original.terminal:break
        finally:original.close()
        c=copy.deepcopy(self.c);c['disturbance']['protocol']='nominal'
        runtime=ReferenceRuntime(c,self.scene,seed=713,episode_id=0)
        LEDGER['native_environment_initializations']+=1
        try:
            for t,expected in enumerate(left):
                row=runtime.transition(public_nominal(runtime.env));LEDGER['native_physical_steps']+=1
                np.testing.assert_allclose(runtime.env.get_agent_state().positions,expected[0],atol=2e-6,rtol=0)
                np.testing.assert_allclose(row['rewards'],expected[1],atol=2e-6,rtol=0)
                self.assertEqual(runtime.terminal,all(expected[2]))
        finally:runtime.close()

    def test_factors_match_native_core_and_memory(self):
        for case in ('force_drag','speed_cap','delay_lag','noise'):
            with self.subTest(case=case):
                env=self.core(clear=True);o=copy.deepcopy(self.c['disturbance']);o['protocol']='episode'
                process=DisturbanceProcess(o,87);x=np.tile(NOMINAL,(4,1))
                if case=='force_drag':
                    x[:,:2]=[.26,.02];x[:,2]=[1.2,.9,1.1,.85];x[:,3]=[.01,-.01,.02,0];x[:,4]=[.85,1.1,1,.95]
                elif case=='speed_cap':
                    x[:,5]=.9
                    env.unwrapped._agent_vel[:,0]=torch.tensor([s['v_xy_max']*.98 for s in env.get_public_vehicle_specs()])
                elif case=='delay_lag':x[:,6]=.25;x[:,7]=2
                else:x[:,8]=.02
                process.current=PrivilegedContext(x);hook=PhysicalIntervention(env.unwrapped,process)
                try:
                    for t in range(3 if case=='delay_lag' else 1):
                        # The native API consumes float32. Compute the independent
                        # actuator oracle from that actual command representation.
                        before=env.get_agent_state();u=np.full((4,3),.4 if t==0 else 0.,dtype=np.float32).astype(float)
                        noise=np.random.default_rng();noise.bit_generator.state=copy.deepcopy(process.noise_rng.bit_generator.state)
                        draw=noise.normal(size=(4,3))
                        expected_u=u.copy()
                        if case=='delay_lag':expected_u=np.zeros((4,3)) if t<2 else np.full((4,3),float(np.float32(.4))*.75)
                        expected_u=np.clip(expected_u+draw*x[:,8,None],-1,1)
                        expected=predict_velocity(before.velocities,expected_u,x,env.get_public_vehicle_specs(),.2,before.positions)
                        env.step(torch.as_tensor(u,dtype=torch.float32));LEDGER['native_physical_steps']+=1
                        np.testing.assert_allclose(hook.last['actuated_normalized'],expected_u,atol=1e-12,rtol=0)
                        np.testing.assert_allclose(env.get_agent_state().velocities,expected,atol=3e-6,rtol=0)
                        np.testing.assert_array_equal([s['a_xy_max'] for s in env.get_public_vehicle_specs()],[1.3,1.,.9,.8])
                finally:hook.restore();env.close()


if __name__=='__main__':unittest.main(verbosity=2)
