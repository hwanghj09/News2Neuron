import numpy as np
import pandas as pd
import pytest

from news_flywire.flywire_graph import demo_circuit
from news_flywire.article_controls import match_inputs, signatures, subset_events


def test_matching_is_exact_on_hops_and_transmitter_and_disjoint():
    c=demo_circuit();c.neurons['top_nt']='acetylcholine'
    a=match_inputs(c,42);b=match_inputs(c,42)
    assert a==b
    targets,controls=a['common_target_ids'],a['common_control_ids']
    assert len(targets)==len(controls)>0
    assert len(set(controls))==len(controls)
    assert not set(controls)&set(c.cells(role='input')+c.cells(role='output'))
    d=signatures(c);keys=['GF_hops','DNp06_hops','top_nt']
    assert np.array_equal(d.loc[targets,keys],d.loc[controls,keys])
    assert a['common_mismatch']['hop_pair_exact_fraction']==1.
    assert a['common_mismatch']['transmitter_match_fraction']==1.


def test_subset_preserves_each_original_event_and_multiplicity():
    e=np.array([[2,0],[2,0],[4,2],[6,1]])
    actual=subset_events(e,[0,2])
    assert np.array_equal(actual,[[2,0],[2,0],[4,1]])
    assert subset_events(e,[]).shape==(0,2)
    with pytest.raises(ValueError,match='Repeated'):
        subset_events(e,[0,0])


def test_missing_comparable_population_fails_instead_of_fabricating_controls():
    c=demo_circuit();c.neurons['top_nt']='gaba'
    c.neurons.loc[c.neurons.role.eq('input'),'top_nt']='acetylcholine'
    with pytest.raises(ValueError,match='No common support'):
        match_inputs(c,5)
