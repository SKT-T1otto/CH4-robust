"""Mechanism evidence, not a learning/performance experiment matrix."""
import copy,json,random,unittest
from pathlib import Path
import numpy as np
import torch
from chapter4_interval_robust.configuration import ROOT,reference_config
from chapter4_interval_robust.provenance import identity
from chapter4_interval_robust.ab_v2.configuration import load,validate,physical_options,evaluation_contract
from chapter4_interval_robust.ab_v2.parameters import *
from chapter4_interval_robust.ab_v2.estimation import History,Estimate,ConditionalEstimator,diagnostics,reconstruct
from chapter4_interval_robust.ab_v2.runtime import ReferenceRuntime,make_scene,Intervention
from chapter4_interval_robust.ab_v2.learning import MultiAgentSAC,Support,lower_tail
from chapter4_interval_robust.ab_v2 import checkpoint
from chapter4_interval_robust.dynamics import Actuator,DisturbanceProcess,predict_velocity

CONFIG_PATH=ROOT/'configs/ch4/experiment_ab_v2.json'
OUTPUT=None
BASE_CONFIG=None
LEDGER={'native_environment_checked':False,'physical_steps':0,'models':{},'historical_weights_loaded':False,'long_training':False}
def config(method='ab_full'):
    if BASE_CONFIG is None:return load(CONFIG_PATH,method,True)
    from chapter4_interval_robust.ab_v2.configuration import mechanism_config
    c=mechanism_config(BASE_CONFIG);c['method']=method;return validate(c)
def ledger(name,m,steps=0,episodes=0):
    LEDGER['models'][name]={'updates':m.updates,'physical_steps':steps,'episodes':episodes}
    assert m.updates<=4 and steps<=8*32 and episodes<=8
def specs():return [dict(a_xy_max=1.,a_z_max=.7,v_xy_max=4.,v_z_max=3.,drag_xy=.1,drag_z=.2,buoyancy_bias=0.)]*4
def fixtures(m,n=8):
    rng=np.random.default_rng(88);c=m.config
    for k in range(n):
        h=np.zeros((8,4,13),np.float32);h[:,:,12]=1
        e=Estimate(c['prior'],h);e.valid[:]=True
        o=np.zeros((4,165),np.float32);o[:,32]=1;o[:,43:61]=e.features();o[:,61:]=h.transpose(1,0,2).reshape(4,104)
        x=np.tile(NOMINAL,(4,1));x[:,0]=rng.uniform(.15,.28);x[:,1]=rng.uniform(.85,1.2,4)
        xi=context(x,np.zeros((4,3)),c['prior'])
        bank=e.bank(rng,c['risk']['samples'],np.ones(4,bool))
        m.remember(dict(observations=o,active=np.ones(4,bool),proposal=np.zeros((4,3),np.float32),reward=1.,next_observations=o.copy(),next_active=np.ones(4,bool),
            context=xi,next_context=xi.copy(),terminated=bool(k%2),truncated=not bool(k%2),scene_bank=bank,next_history=h,privileged_target=x))

class Contracts(unittest.TestCase):
    def test_parameter_identity_and_unidentifiable_legacy_pair(self):
        b=config()['prior'];x=np.tile(NOMINAL,(4,1));legacy=to_legacy(x,.01)
        np.testing.assert_array_equal(from_legacy(legacy),x);np.testing.assert_array_equal(expand(joint(x)),x)
        self.assertEqual(context(x,np.zeros((4,3)),b).shape,(33,))
        q=legacy.copy();q[:,1]+=.01;q[:,3]-=.01
        np.testing.assert_allclose(predict_velocity(np.zeros((4,3)),np.ones((4,3))*.1,legacy,specs(),.2,np.zeros((4,3))),
                                   predict_velocity(np.zeros((4,3)),np.ones((4,3))*.1,q,specs(),.2,np.zeros((4,3))))
        with self.assertRaises(ValueError):from_legacy(q)
        x[1,0]+=.01
        with self.assertRaises(ValueError):joint(x)
    def test_history_status_and_public_shape(self):
        h=History();self.assertTrue(np.all(h.arrays()[:,:,12]==-1))
        zero=np.zeros((4,3));h.append(zero,zero,zero,zero,[True,False,True,False]);a=h.arrays()
        np.testing.assert_array_equal(a[-1,:,12],[1,0,1,0]);self.assertTrue(np.all(a[:-1,:,12]==-1))
        with self.assertRaises(ValueError):PolicyInput(np.zeros((4,61)),np.ones(4,bool))
    def test_delay_lag_pulse_and_reset(self):
        act=Actuator(2);x=to_legacy(np.tile(NOMINAL,(4,1)));x[:,7]=2;x[:,6]=.25;zero=np.zeros((4,3))
        np.testing.assert_array_equal(act.apply(np.ones((4,3)),x,zero),zero)
        np.testing.assert_array_equal(act.apply(zero,x,zero),zero)
        np.testing.assert_allclose(act.apply(zero,x,zero),.75)
        np.testing.assert_allclose(act.apply(zero,x,zero),.1875)
        act.reset();np.testing.assert_array_equal(act.filtered,zero)
    def test_timing_eliminates_wrong_delay(self):
        c=config();opts=c['estimation'].copy();opts.update(minimum_candidates=1,minimum_data=1,replenish=False,velocity_tolerance=1e-6,padding=0)
        e=ConditionalEstimator(c['prior'],specs(),.2,opts,8);e.pool[:]=NOMINAL
        K=e.pool.shape[2];e.pool[:,:,:K//2,5]=1;e.pool[:,:,K//2:,5]=0
        truth=np.tile(NOMINAL,(4,1));truth[:,5]=1;act=Actuator(2);p=np.zeros((4,3));v=p.copy()
        for t,u in enumerate([.8,-.5,.3,-.2]):
            command=np.full((4,3),u);applied=act.apply(command,to_legacy(truth),np.zeros((4,3)))
            nv=predict_velocity(v,applied,to_legacy(truth),specs(),.2,p)
            result=e.update(p,v,command,nv,np.ones(4,bool));p=p+nv*.2;v=nv
        self.assertTrue(result.valid.all());np.testing.assert_array_equal(result.delay_mask,[[False,True,False]]*4)
        samples,z,info=result.samples(np.random.default_rng(2),20)
        np.testing.assert_array_equal(samples[:,:,5],1);np.testing.assert_allclose(z,np.broadcast_to(act.filtered,z.shape),atol=1e-12)
    def test_shared_conditions_and_individual_responses_not_intersected(self):
        c=config();opts=c['estimation'].copy();opts.update(minimum_candidates=1,minimum_data=1,replenish=False,velocity_tolerance=1e-7,padding=0)
        e=ConditionalEstimator(c['prior'],specs(),.2,opts,5);e.pool[:]=NOMINAL
        e.pool[:,:,:,0]=np.linspace(.15,.28,e.pool.shape[0])[:,None,None]
        e.pool[:,:,:,2]=np.array([.85,.95,1.,1.1])[None,:,None]
        truth=e.pool[-1,:,0].copy();zero=np.zeros((4,3));u=np.ones((4,3))*.3
        nv=predict_velocity(zero,u,to_legacy(truth),specs(),.2,zero)
        estimate=e.update(zero,zero,u,nv,np.ones(4,bool));x,_,_=estimate.samples(np.random.default_rng(7),16)
        np.testing.assert_array_equal(x[:,:,0],truth[0,0]);np.testing.assert_array_equal(x[:,:,2],np.broadcast_to(truth[:,2],(16,4)))
        self.assertTrue(diagnostics(estimate,truth,opts)['conditional_support'])
    def test_empty_set_and_unobserved_vehicle_are_honest(self):
        c=config();e=ConditionalEstimator(c['prior'],specs(),.2,c['estimation'],4);zero=np.zeros((4,3))
        r=e.update(zero,zero,zero,np.full((4,3),100.),np.zeros(4,bool))
        self.assertFalse(r.valid.any());self.assertEqual(r.support.sum(),0)
        e.update(zero,zero,zero,np.full((4,3),100.),np.ones(4,bool))
        r=e.update(zero,zero,zero,np.full((4,3),100.),np.ones(4,bool))
        self.assertFalse(r.valid.any());self.assertIn('reset',r.reason)
        np.testing.assert_array_equal(r.lower,np.tile(np.asarray(c['prior'])[:,0],(4,1)))
    def test_same_world_and_preserved_filter_bank(self):
        c=config();e=Estimate(c['prior']);e.valid[:]=True
        bank=e.bank(np.random.default_rng(2),32,np.ones(4,bool));a=np.array(bank['contexts'])
        for row in a:self.assertEqual(expand(row[:21]).shape,(4,6))
        self.assertEqual(bank['schema'],'ch4.conditional_scene_bank33.v2')
        np.testing.assert_allclose(a[:,21:],.5)
    def test_configuration_ablation_and_split_contract(self):
        c=config();self.assertEqual(c['method_options']['estimator'],config('b_off')['method_options']['estimator'])
        self.assertFalse(config('b_off')['method_options']['aligned']);self.assertFalse(config('a_off')['method_options']['shared'])
        self.assertTrue(config('width_off')['method_options']['width_off']);self.assertEqual(config('c_off')['method_options']['risk_alpha'],1)
        bad=copy.deepcopy(c);bad['noise_std']=.02
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(c);bad['prior'][5][1]=3
        with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(c);bad['seed']+=1
        with self.assertRaises(ValueError):evaluation_contract(c,bad)
        scenes=[make_scene(c,s,0) for s in ('train','calib','dev','test')]
        self.assertEqual(len({s['disturbance_seed'] for s in scenes}),4)
        self.assertEqual(make_scene(c,'dev',0)['change_steps'],make_scene(c,'dev',0)['change_steps'])
    def test_fixed_prior_is_valid_and_stale_data_is_not_fresh(self):
        c=config();e=ConditionalEstimator(c['prior'],specs(),.2,c['estimation'],3)
        zero=np.zeros((4,3));u=np.zeros((4,3))
        for _ in range(2):
            nv=predict_velocity(zero,u,to_legacy(np.tile(NOMINAL,(4,1))),specs(),.2,zero)
            e.update(zero,zero,u,nv,np.ones(4,bool))
        age=e.update(zero,zero,u,nv,np.zeros(4,bool)).age
        self.assertTrue(np.all(age>=1))
    def test_tail_direction_and_joint_support(self):
        q=torch.tensor([[4.,-3.,2.,-1.]],requires_grad=True);value=lower_tail(q,.5,2);value.backward()
        self.assertEqual(float(value.detach()),-2);np.testing.assert_array_equal(q.grad,[[0,.5,0,.5]])
        c=config();m=MultiAgentSAC(c);fixtures(m,8);gate=m.support
        row=m.replay.rows[0];query=np.tile(row['context'],(1,2,1));o=np.array([row['observations']]);a=np.array([row['proposal']])
        # Broad explicit fixture radii isolate the whole-query support rule.
        gate.opts=copy.deepcopy(gate.opts);gate.opts['minimum_facts']=2
        gate.opts['radii']={k:10 for k in gate.opts['radii']}
        result=gate.supported(o,a,query);self.assertTrue(result.all())
        query[:,1,:21]=100;self.assertFalse(gate.supported(o,a,query).all(1)[0])
        ledger('support',m)
    def test_sac_bootstrap_polyak_and_actor_only_mask(self):
        c=config('width_off');m=MultiAgentSAC(c);fixtures(m,8)
        with torch.no_grad():
            m.log_temperature.fill_(-40)
            for p in m.target.parameters():p.zero_()
            m.target.q1[-1].bias.fill_(2);m.target.q2[-1].bias.fill_(2)
        old=copy.deepcopy(m.target.state_dict());metrics=m.update()
        # Replay sampling may select different terminal counts; derive from its
        # recorded bootstrap count rather than assuming an alternating batch.
        expected=1+.95*2*metrics['bootstrap_rows']/c['sac']['batch_size']
        self.assertAlmostEqual(metrics['target_mean'],expected,places=5)
        for k,v in m.target.state_dict().items():torch.testing.assert_close(v,old[k]*(1-c['sac']['tau'])+m.critic.state_dict()[k]*c['sac']['tau'])
        original=torch.tensor(np.array([m.replay.rows[0]['observations']]));before=original.clone();seen=[]
        handle=m.actor.agents[0][0].register_forward_pre_hook(lambda module,args:seen.append(args[0].detach().clone()))
        m.actor.sample(original,torch.ones((1,4),dtype=torch.bool),True);handle.remove()
        torch.testing.assert_close(original,before);torch.testing.assert_close(seen[0][:,49:55],torch.ones((1,6)))
        ledger('sac',m)
    def test_checkpoint_boundary_exact_next_update_and_v1_rejection(self):
        c=config();m=MultiAgentSAC(c);fixtures(m,8);m.update();path=OUTPUT/'unit_checkpoint.pt'
        checkpoint.save(path,m,c,1,[{'mechanism_fixture':True}]);payload=checkpoint.load(path,c)
        expected=m.update();weights=copy.deepcopy(m.actor.state_dict())
        restored=MultiAgentSAC(c);checkpoint.restore(payload,restored);actual=restored.update();self.assertEqual(actual,expected)
        for k,v in weights.items():torch.testing.assert_close(restored.actor.state_dict()[k],v,rtol=0,atol=0)
        bad=OUTPUT/'fake_v1.pt';bad.write_bytes(b'never deserialize');bad.with_suffix('.json').write_text(json.dumps({'schema':'auv.ch4.episode_boundary_checkpoint.v1'}))
        with self.assertRaises(ValueError):checkpoint.load(bad,c)
        ledger('checkpoint_original',m);ledger('checkpoint_restored',restored)
    def test_risk_branch_uses_low_tail_and_rejects_unsupported_scene(self):
        class AnalyticQ(torch.nn.Module):
            def __init__(self):super().__init__();self.scale=torch.nn.Parameter(torch.tensor(1.))
            def forward(self,o,a,xi):
                q=xi[:,0]*self.scale+a.flatten(1).sum(-1)*0
                return q,q
        for name,alpha in [('ab_full',.25),('c_off',1.)]:
            c=config(name);m=MultiAgentSAC(c);fixtures(m,8)
            m.critic=AnalyticQ();m.target=AnalyticQ().requires_grad_(False)
            m.critic_optimizer=torch.optim.Adam(m.critic.parameters(),lr=0.)
            m.support.opts=copy.deepcopy(m.support.opts);m.support.opts['minimum_facts']=2
            m.support.opts['radii']={k:10 for k in m.support.opts['radii']}
            for row in m.replay.rows:
                row['scene_bank']['valid']=True
                for k,xi in enumerate(row['scene_bank']['contexts']):xi[0]=k/31
            first=m.update();self.assertEqual(first['risk_supported_rows'],4)
            self.assertAlmostEqual(first['actor_utility_mean'],(int(alpha*32)-1)/62,places=6)
            for row in m.replay.rows:row['scene_bank']['contexts'][-1][0]=100
            second=m.update();self.assertEqual(second['risk_supported_rows'],0)
            self.assertEqual(second['risk_fallback_rows'],4)
            ledger('analytic_risk_'+name,m)
    def test_set_summary_support_and_supervised_reference(self):
        c=config('sirsa_naive');m=MultiAgentSAC(c);fixtures(m,8)
        m.support.opts=copy.deepcopy(m.support.opts);m.support.opts['minimum_facts']=2
        m.support.opts['radii']={k:10 for k in m.support.opts['radii']};m.support.opts['radii']['set']=.1
        row=m.replay.rows[0];o=np.array([row['observations']]);a=np.array([row['proposal']]);q=np.tile(row['context'],(1,2,1))
        self.assertTrue(m.support.supported(o,a,q).all());o[:,:,43:61]=100
        self.assertFalse(m.support.supported(o,a,q).any())
        frame=row['next_history'];untrained=m.identifier.infer(frame)
        self.assertFalse(untrained.valid.any());update=m.update()
        self.assertTrue(np.isfinite(update['identifier_loss']));self.assertEqual(m.identifier.updates,1)
        frozen=copy.deepcopy(m.identifier.state_dict());trained=m.identifier.infer(frame)
        self.assertTrue(trained.valid.all())
        for k,v in frozen.items():torch.testing.assert_close(m.identifier.state_dict()[k],v,rtol=0,atol=0)
        ledger('supervised_reference',m)
    def test_factorial_kernel_budget_is_matched(self):
        c=config();counts=[];zero=np.zeros((4,3))
        for shared in (False,True):
            for aligned in (False,True):
                e=ConditionalEstimator(c['prior'],specs(),.2,c['estimation'],7,shared=shared,aligned=aligned)
                e.update(zero,zero,zero,zero,np.zeros(4,bool));counts.append(e.prediction_calls)
        self.assertEqual(len(set(counts)),1)
        # Different reset/evidence lengths must not change the charged kernel
        # budget, and padding must never create valid data support.
        counts=[]
        for length in (0,3,7):
            e=ConditionalEstimator(c['prior'],specs(),.2,c['estimation'],7)
            for _ in range(length):e.history.append(zero,zero,zero,zero,np.zeros(4,bool));e.commands.append(zero)
            result=e.update(zero,zero,zero,zero,np.zeros(4,bool));counts.append(e.prediction_calls)
            self.assertEqual(result.support.sum(),0)
            self.assertEqual(e.padding_prediction_calls,(7-length)*2*np.prod(e.pool.shape[:-1]))
        self.assertEqual(len(set(counts)),1)

class Native(unittest.TestCase):
    def test_original_D2_reference_equivalence(self):
        from chapter3_bser.learning.runtime import RepairedMissionRuntime
        from chapter4_interval_robust.runtime import public_nominal
        class Zero:
            def step(self,observations,explore=True):return [torch.zeros((1,3)) for _ in range(4)]
        c=config('nominal');c['disturbance']['protocol']='nominal';scene=make_scene(c,'dev',0)
        original=RepairedMissionRuntime(reference_config(c),scene,seed=713,episode_id=0);expected=[]
        try:
            for _ in range(4):
                row=original.advance(Zero(),explore=False);LEDGER['physical_steps']+=1
                expected.append((np.asarray(original.env.get_agent_state().positions),np.asarray(row['rewards'])))
        finally:original.close()
        runtime=ReferenceRuntime(c,scene,seed=713,episode_id=0)
        try:
            for p,rewards in expected:
                row=runtime.transition(public_nominal(runtime.env));LEDGER['physical_steps']+=1
                np.testing.assert_allclose(runtime.env.get_agent_state().positions,p,atol=2e-6,rtol=0)
                np.testing.assert_allclose(row['rewards'],rewards,atol=2e-6,rtol=0)
        finally:runtime.close()
    def test_reduced_factors_against_native_integrator(self):
        from chapter3_bser.learning.base_runtime import _make_base_env,seed_innovations
        from chapter4_interval_robust.parameters import PrivilegedContext
        c=config();scene=make_scene(c,'dev',3);scene.update(obstacles=[],use_obstacles=False,obstacle_count=0,obstacle_layout_id='none')
        for mode in ('response','delay_lag'):
            seed_innovations(771);env=_make_base_env(reference_config(c));env.reset(scenario=copy.deepcopy(scene))
            env.unwrapped.use_residual_prior=False;env.unwrapped._residual_scale.fill_(1.)
            opts=physical_options(c,scene);opts['protocol']='episode';process=DisturbanceProcess(opts,71)
            truth=np.tile(NOMINAL,(4,1));truth[:,0]=.26
            if mode=='response':
                truth[:,1]=[1.2,.9,1.1,.85];truth[:,2]=[.85,1.1,1.,.95];truth[:,3]=[.01,-.01,.02,0]
            else:truth[:,4]=.25;truth[:,5]=2
            process.current=PrivilegedContext(to_legacy(truth,c['noise_std']));hook=Intervention(env.unwrapped,process)
            try:
                for t in range(3):
                    state=env.get_agent_state();u=np.full((4,3),.4 if t==0 else 0,dtype=np.float32)
                    noise=np.random.default_rng();noise.bit_generator.state=copy.deepcopy(process.noise_rng.bit_generator.state)
                    expected=u.astype(float) if mode=='response' else np.full((4,3),0 if t<2 else float(u.dtype.type(.4))*.75)
                    expected=np.clip(expected+noise.normal(size=(4,3))*c['noise_std'],-1,1)
                    prediction=predict_velocity(state.velocities,expected,to_legacy(truth,c['noise_std']),env.get_public_vehicle_specs(),env.get_public_control_contract()['dt'],state.positions)
                    pre=process.actuator.filtered.copy();env.step(torch.tensor(u));LEDGER['physical_steps']+=1
                    np.testing.assert_array_equal(hook.last['pre_filter'],pre)
                    np.testing.assert_allclose(hook.last['actuated_normalized'],expected,atol=1e-12)
                    np.testing.assert_allclose(env.get_agent_state().velocities,prediction,atol=3e-6,rtol=0)
            finally:hook.restore();env.close()
    def test_nominal_pair_rng_and_native_factors(self):
        from chapter3_bser.learning.base_runtime import _make_base_env,seed_innovations
        c=config();scene=make_scene(c,'dev',0);traces=[]
        for hooked in (False,True):
            seed_innovations(772);env=_make_base_env(reference_config(c));env.reset(scenario=copy.deepcopy(scene))
            env.unwrapped.use_residual_prior=False;env.unwrapped._residual_scale.fill_(1.)
            hook=None;rows=[]
            if hooked:
                opts=physical_options(c,scene);opts['protocol']='nominal';hook=Intervention(env.unwrapped,DisturbanceProcess(opts,55))
            try:
                for step in range(4):
                    o,r,d=env.step(torch.full((4,3),.1));LEDGER['physical_steps']+=1
                    rows.append((np.asarray(o).copy(),np.asarray(r).copy(),d,torch.get_rng_state().clone(),copy.deepcopy(np.random.get_state()),random.getstate()))
                traces.append(rows)
            finally:
                if hook:hook.restore()
                env.close()
        for left,right in zip(*traces):
            np.testing.assert_array_equal(left[0],right[0]);np.testing.assert_array_equal(left[1],right[1]);self.assertEqual(left[2],right[2]);self.assertTrue(torch.equal(left[3],right[3]));self.assertEqual(left[5],right[5])
            for a,b in zip(left[4],right[4]):np.testing.assert_array_equal(a,b)
        LEDGER['native_environment_checked']=True
    def test_runtime_context_timing_and_information_boundary(self):
        c=config();scene=make_scene(c,'dev',1);runtime=ReferenceRuntime(c,scene,seed=713,episode_id=0)
        m=MultiAgentSAC(c)
        try:
            for t in range(4):
                before=runtime.policy_input();row=runtime.transition(np.full((4,3),.2,np.float32));LEDGER['physical_steps']+=1
                np.testing.assert_array_equal(row['observations'],before.observations)
                np.testing.assert_allclose(row['context'][21:].reshape(4,3)*2-1,row['privileged_pre_filter'],atol=1e-7)
                self.assertEqual(row['next_history'][-1,:,12].min()>=0,True)
                self.assertTrue(np.all(row['privileged_theta'][:,0]==row['privileged_theta'][0,0]))
                clean=runtime.policy_input().observations.copy();runtime.intervention.last['truth']='must never read'
                np.testing.assert_array_equal(runtime.policy_input().observations,clean)
            with self.assertRaises(TypeError):m.act({'observations':clean,'truth':1})
        finally:runtime.close()
        ledger('runtime_input',m,4,1)
    def test_all_baseline_and_ablation_runtime_interfaces(self):
        for name in ('s0t0','s1t0','s0t1','ab_full','uniform_dr','point','fixed_set','sirsa_naive','c_off','width_off','memory_off','nominal'):
            c=config(name);scene=make_scene(c,'dev',2);m=MultiAgentSAC(c) if c['method_options']['learn'] else None
            runtime=ReferenceRuntime(c,scene,seed=713,episode_id=0,identifier=None if m is None else m.identifier)
            try:
                for t in range(2):
                    public=runtime.policy_input();u=np.zeros((4,3)) if m is None else m.act(public,True)
                    row=runtime.transition(u);LEDGER['physical_steps']+=1
                    self.assertTrue(np.isfinite(row['next_observations']).all())
            finally:runtime.close()
            if m is not None:ledger('interface_'+name,m,2,1)
