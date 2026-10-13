import json
from copy import deepcopy

import pytest
from langchain_core.runnables import RunnableLambda
from test_dynamic_investigation import Environment, registry, unknown_action
from test_engineering_alternatives import DiscoveryV7

from rca_agent.core.dynamic import MechanismDraft, VerificationPlanPayload, freeze_mechanism
from rca_agent.inquiry_policy import InquiryInvestigationPolicy
from rca_agent.langchain_acquisition import InquiryHandoff, LangChainInquiryAdapter, query_options
from rca_agent.models import RequestStopAction
from rca_agent.models.contract import MetricDescriptor
from rca_agent.runner import EpisodeRunner, RecordedReplay
from rca_agent.runner.transcript import EventType, Transcript
from rca_agent.tools.fingerprint import sha256_digest


class PublicEnvironment(Environment):
    def reset(self):
        base = super().reset()
        cap = base.capabilities[0].model_copy(update={'metrics': tuple(
            MetricDescriptor(entity_type='service', signal=s, unit='ratio', description='Measured occupancy')
            for s in ('active_ratio', 'queue_ratio'))})
        return base.model_copy(update={'capabilities': (cap,)})

class EmptyPlanner:
    def plan(self, context):
        return VerificationPlanPayload(plans=())

def setup(mode='active', version=7):
    env = PublicEnvironment()
    runner = EpisodeRunner(env, evidence_registry=registry(), discovery=DiscoveryV7(),
        verification_planner=EmptyPlanner(), investigation_workspace_mode=mode, bound_handoff_version=version)
    bootstrap = runner.start()
    options = query_options({'bootstrap': bootstrap.model_dump(mode='json')},
                            [bootstrap.alert_entity.model_dump(mode='json')])
    return env, runner, options

def output(payload, pairs=(("n1", "q1"), ("n1", "q2"))):
    menu = json.loads(payload['context'])['queries']
    return {'hypotheses': [{'hypothesis_id':'h1','mechanism':'worker exhaustion',
                           'alternative':'connection pool exhaustion'}],
        'needs': [{'need_id': n, 'hypothesis_id':'h1', 'question':'Measure occupancy',
                   'expected_if_true':'busy', 'expected_if_alternative':'not busy',
                   'entity_ref':'ent-service','observable_signals':['active_ratio','queue_ratio']}
                  for n in dict.fromkeys(n for n,q in pairs)],
        'choices':[{'need_id':n,'query_ref':q,'mode':'direct','purpose':'Measure occupancy',
                    'measurement_target':{k:menu[q][k] for k in ('entity_ref','signal','measurement_kind','windows')}}
                   for n,q in pairs]}

def policy_for(options, pairs, mutation=None, limit=2):
    def respond(payload):
        raw=output(payload,pairs)
        if mutation: mutation(raw)
        return raw
    policy=InquiryInvestigationPolicy(LangChainInquiryAdapter(RunnableLambda(respond)),query_limit=limit)
    policy.selection_calls=1
    policy._plan_inquiry_round({},options)
    return policy

def execute(runner,policy):
    runner.record_inquiry_round(policy.inquiry_drafts[-1])
    history=[]
    for action in policy.selection_queue:
        routed=runner.route_active_inquiry_action(action)
        if routed is None: continue
        response=runner.handle(routed)
        history.append({'action':routed.model_dump(mode='json'),
                        'result':{'kind':'tool_response','value':response.model_dump(mode='json')}})
    return history

@pytest.mark.parametrize('pairs,calls,links',[( (('n1','q1'),('n1','q2')),2,2),
                                            ( (('n1','q1'),('n2','q1')),1,2)])
def test_independent_full_binding_chain(pairs,calls,links):
    env,runner,options=setup()
    policy=policy_for(options,pairs)
    history=execute(runner,policy)
    handoff=policy.discovery_handoff(tuple(history))
    assert len(env.calls)==calls and env.remaining==20-calls
    assert len(handoff['query_links'])==links
    assert len(runner.investigation_workspace_snapshot.tasks)==links
    assert all(t.attempts and t.attempts[0].completed_sequence for t in runner.investigation_workspace_snapshot.tasks)
    runner.handle(unknown_action())
    runner.discover_candidates(inquiry_handoff=handoff)
    runner.handle(RequestStopAction(summary='independent binding check'))
    assert runner.answer.verify_digest() and RecordedReplay.verify(runner.transcript.events).valid

@pytest.mark.parametrize('pairs',[(('n1','q1'),('n1','q2')), (('n1','q1'),('n2','q1'))])
def test_independent_budget_prefix_does_not_misbind(pairs):
    env,runner,options=setup()
    policy=policy_for(options,pairs,limit=1)
    history=execute(runner,policy)
    assert env.calls==['active_ratio']
    links=policy.discovery_handoff(tuple(history))['query_links']
    assert len(links)==(2 if pairs[1][1]=='q1' else 1)
    assert all(l['signal']=='active_ratio' for l in links)

@pytest.mark.parametrize('field',['mode','purpose','measurement_target'])
def test_missing_required_choice_is_rejected(field):
    _,_,options=setup()
    from rca_agent.inquiry_policy import InquiryUnavailableError
    with pytest.raises(InquiryUnavailableError):
        policy_for(options,(('n1','q1'),),lambda r:r['choices'][0].pop(field))

def test_extra_choice_field_must_match_live_schema():
    _,_,options=setup()
    from rca_agent.inquiry_policy import InquiryUnavailableError
    with pytest.raises(InquiryUnavailableError):
        policy_for(options,(('n1','q1'),),lambda r:r['choices'][0].update(unexpected='ignored'))

@pytest.mark.parametrize('mode',['shadow','active','off'])
def test_unknown_version_must_be_rejected_by_runner(mode):
    _,runner,options=setup(mode)
    policy=policy_for(options,(('n1','q1'),))
    record=deepcopy(policy.inquiry_drafts[0])
    record['draft']['contract_version']=999
    record['draft']['needs'][0]['observable_signals']=['wrong_signal']
    with pytest.raises(ValueError): runner.record_inquiry_round(record)

@pytest.mark.parametrize('window',['same','wider','narrower','empty'])
def test_explicit_window_boundaries(window):
    _,_,options=setup()
    def mutation(raw):
        from datetime import timedelta

        from rca_agent.models import TimeWindow
        w=TimeWindow.model_validate(raw['choices'][0]['measurement_target']['windows']['series'])
        if window=='wider': w=w.model_copy(update={'start':w.start-timedelta(seconds=1)})
        if window=='narrower': w=w.model_copy(update={'start':w.start+timedelta(seconds=1)})
        raw['needs'][0]['windows']={} if window=='empty' else {'series':w.model_dump(mode='json')}
    from rca_agent.inquiry_policy import InquiryUnavailableError
    if window=='narrower':
        with pytest.raises(InquiryUnavailableError): policy_for(options,(('n1','q1'),),mutation)
    else:
        assert policy_for(options,(('n1','q1'),),mutation)

@pytest.mark.parametrize('self_second',[False,True])
def test_external_colliding_alternative_ids_never_delete_original(self_second):
    _,runner,_=setup('off')
    from test_dynamic_investigation import query
    runner.handle(query('active_ratio'))
    runner.handle(unknown_action())
    handoff=InquiryHandoff.model_validate({'revision':1,'hypotheses':[
        {'hypothesis_id':'h1','mechanism':'worker exhaustion','alternative':'connection pool exhaustion','alternative_id':'alt-shared'},
        {'hypothesis_id':'h2','mechanism':'worker exhaustion','alternative':
         ' WORKER  exhaustion ' if self_second else 'frontend worker exhaustion','alternative_id':'alt-shared'}],
        'needs':[],'query_links':[]})
    runner.discover_candidates(inquiry_handoff=handoff.model_dump(mode='json'))
    freezes=[e.payload['mechanism'] for e in runner.transcript.events if e.event_type==EventType.MECHANISM_DRAFT_FROZEN]
    runner.handle(RequestStopAction(summary='external identity collision'))
    assert RecordedReplay.verify(runner.transcript.events).valid
    if self_second:
        assert runner.answer.inquiry_candidates[1].discovery_disposition=='deferred'
    else:
        assert (not freezes or len(freezes[0]['alternatives'])==2
                or runner.answer.inquiry_candidates[1].discovery_disposition=='deferred')

def test_later_need_cannot_borrow_response_issued_before_declaration():
    env,runner,options=setup()
    policy=policy_for(options,(('n1','q1'),))
    history=execute(runner,policy)
    policy.selection_calls=2
    policy.inquiry.runnable=RunnableLambda(lambda p:output(p,(('n2','q1'),)))
    policy._plan_inquiry_round({},options)
    runner.record_inquiry_round(policy.inquiry_drafts[-1])
    assert runner.route_active_inquiry_action(policy.selection_queue[0]) is None
    assert len(env.calls)==1
    handoff=policy.discovery_handoff(tuple(history))
    runner.handle(unknown_action())
    runner.discover_candidates(inquiry_handoff=handoff)
    runner.handle(RequestStopAction(summary='late binding check'))
    assert RecordedReplay.verify(runner.transcript.events).valid
    assert not any(l['source_round']==2 for l in handoff['query_links'])

@pytest.mark.parametrize('tamper',['text','provenance','id'])
def test_frozen_alternative_tamper_rejected_after_rehash(tamper):
    _,runner,options=setup('off')
    policy=policy_for(options,(('n1','q1'),))
    history=execute(runner,policy)
    runner.handle(unknown_action())
    runner.discover_candidates(inquiry_handoff=policy.discovery_handoff(tuple(history)))
    runner.handle(RequestStopAction(summary='freeze tamper check'))
    assert RecordedReplay.verify(runner.transcript.events).valid
    freeze=next(e for e in runner.transcript.events if e.event_type==EventType.MECHANISM_DRAFT_FROZEN)
    raw=deepcopy(freeze.payload['mechanism'])
    if tamper=='text': raw['alternatives'][0]['mechanism']='different competing explanation'
    elif tamper=='id': raw['alternatives'][0]['alternative_id']='alt-replaced'
    else: raw['alternatives'][0]['inquiry_provenance']['hypothesis_id']='h-forged'
    changed=freeze_mechanism(MechanismDraft.model_validate({k:v for k,v in raw.items() if k not in {'digest','mechanism_id'}})).model_dump(mode='json')
    replacements={raw['mechanism_id']:changed['mechanism_id'],raw['digest']:changed['digest']}
    def replace(x):
        if isinstance(x,dict): return {k:replace(v) for k,v in x.items()}
        if isinstance(x,list): return [replace(v) for v in x]
        if isinstance(x,str):
            for a,b in replacements.items(): x=x.replace(a,b)
        return x
    rebuilt=Transcript(runner.episode_id)
    for event in runner.transcript.events:
        payload=replace(deepcopy(event.payload))
        if event==freeze: payload['mechanism']=changed
        if event.event_type==EventType.ANSWER_FROZEN:
            payload['transcript_digest']=rebuilt.events[-1].digest
            payload['artifact_digest']=sha256_digest({k:v for k,v in payload.items() if k!='artifact_digest'})
        rebuilt.append(event.event_type,payload)
    with pytest.raises(ValueError) as rejected: RecordedReplay.verify(rebuilt.events)
    print('Frozen tamper rejection:',tamper,str(rejected.value))

def test_legacy_host_rollback_is_consistent():
    from test_bound_formalization import BoundDiscovery
    from test_dynamic_investigation import TwoStagePlanner, query
    env=PublicEnvironment()
    runner=EpisodeRunner(env,discovery=BoundDiscovery(),verification_planner=TwoStagePlanner(),
                         evidence_registry=registry(),bound_handoff_version=6)
    bootstrap=runner.start()
    options=query_options({'bootstrap':bootstrap.model_dump(mode='json')},[bootstrap.alert_entity.model_dump(mode='json')])
    def legacy(payload):
        raw=output(payload,(('n1','q1'),))
        for key in ('mode','purpose','measurement_target'):raw['choices'][0].pop(key)
        return raw
    adapter=LangChainInquiryAdapter(RunnableLambda(legacy),contract_version=1)
    parsed,_actions=adapter.plan({},options)
    assert parsed.contract_version is None
    assert all(key not in parsed.model_dump()['choices'][0] for key in ('mode','purpose','measurement_target'))
    runner.handle(query('active_ratio'))
    runner.handle(unknown_action())
    handoff={'revision':1,'hypotheses':[{'hypothesis_id':'h1','mechanism':'worker exhaustion',
                                      'alternative':'worker exhaustion'}], 'needs':[],'query_links':[]}
    runner.discover_candidates(inquiry_handoff=handoff)
    runner.handle(RequestStopAction(summary='explicit legacy host rollback'))
    assert any(e.event_type==EventType.MECHANISM_DRAFT_FROZEN for e in runner.transcript.events)
    assert RecordedReplay.verify(runner.transcript.events).valid

@pytest.mark.parametrize('change',['binding_id','mode','action','target'])
def test_independent_binding_tamper_rejected_with_rebound_answer(change):
    _,runner,options=setup('off')
    policy=policy_for(options,(('n1','q1'),))
    execute(runner,policy)
    runner.handle(RequestStopAction(summary='binding tamper check'))
    rebuilt=Transcript(runner.episode_id)
    for event in runner.transcript.events:
        payload=deepcopy(event.payload)
        if event.event_type==EventType.INQUIRY_ROUND_RECORDED:
            binding=payload['selected_actions'][0]
            if change=='action': binding['action']['request']['signal']='queue_ratio'
            elif change=='target':binding['measurement_target']['signal']='queue_ratio'
            else:binding[change]='exploratory' if change=='mode' else 'binding-forged'
        if event.event_type==EventType.ANSWER_FROZEN:
            payload['transcript_digest']=rebuilt.events[-1].digest
            payload['artifact_digest']=sha256_digest({k:v for k,v in payload.items() if k!='artifact_digest'})
        rebuilt.append(event.event_type,payload)
    with pytest.raises(ValueError,match='invalid Inquiry contract bindings'):
        RecordedReplay.verify(rebuilt.events)

@pytest.mark.parametrize('mode',['shadow','active'])
def test_direct_mismatch_cannot_execute_exploratory_has_no_direct_coverage(mode):
    env,runner,options=setup(mode)
    from rca_agent.inquiry_policy import InquiryUnavailableError
    def wrong(raw):raw['needs'][0]['observable_signals']=['unavailable_sql_signal']
    with pytest.raises(InquiryUnavailableError):policy_for(options,(('n1','q1'),),wrong)
    assert env.calls==[]
    def explore(raw):
        wrong(raw)
        raw['choices'][0]['mode']='exploratory'
    policy=policy_for(options,(('n1','q1'),),explore)
    execute(runner,policy)
    task=runner.investigation_workspace_snapshot.tasks[0]
    assert task.scheduling['alignment']=='exploratory'
    assert not any('COVERAGE' in reason for attempt in task.attempts for reason in attempt.selection_reason_codes)
    runner.handle(RequestStopAction(summary='exploratory direct separation'))
    assert RecordedReplay.verify(runner.transcript.events).valid

@pytest.mark.parametrize('kind',['long','capacity','unknown'])
def test_alternative_representation_limits_remain_visible(kind):
    _,runner,_options=setup('off')
    from test_dynamic_investigation import query
    runner.handle(query('active_ratio'))
    runner.handle(unknown_action())
    texts=(['x'*501] if kind=='long' else [f'other causal explanation {i}' for i in range(9)]
           if kind=='capacity' else ['unclassified competing explanation'])
    handoff={'revision':1,'hypotheses':[{'hypothesis_id':f'h{i}','mechanism':'worker exhaustion',
                                       'alternative':text} for i,text in enumerate(texts)],'needs':[],'query_links':[]}
    runner.discover_candidates(inquiry_handoff=handoff)
    runner.handle(RequestStopAction(summary='limits preserve originals'))
    assert [c.alternative for c in runner.answer.inquiry_candidates]==texts
    if kind!='unknown':
        assert not any(e.event_type==EventType.MECHANISM_DRAFT_FROZEN for e in runner.transcript.events)
        assert all(c.discovery_disposition=='deferred' for c in runner.answer.inquiry_candidates)
    else:
        frozen=next(e for e in runner.transcript.events if e.event_type==EventType.MECHANISM_DRAFT_FROZEN)
        assert frozen.payload['mechanism']['alternatives'][0]['mechanism_class']=='unknown'
    assert RecordedReplay.verify(runner.transcript.events).valid

def test_menu_discloses_actual_public_result_bound():
    _,_,options=setup()
    received=[]
    def respond(payload):
        received.append(json.loads(payload['context'])['queries'])
        return output(payload,(('n1','q1'),))
    LangChainInquiryAdapter(RunnableLambda(respond)).plan({},options)
    action=next(iter(options.values()))['action']
    assert action.request.max_points==100
    # Bound must appear somewhere in the model-visible measurement description;
    # do not prescribe a new field name. The synthetic menu contains no unrelated 100.
    assert str(action.request.max_points) in json.dumps(received[0]['q1'])
