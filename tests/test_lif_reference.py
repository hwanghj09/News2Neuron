"""Independent numerical cross-check against Brian2 for a deterministic small circuit."""
import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from news_flywire.flywire_graph import Circuit
from news_flywire.brain_simulation import simulate


def test_reduced_lif_transmission_agrees_with_brian2():
    b=pytest.importorskip("brian2")
    b.start_scope()
    b.prefs.codegen.target="numpy"
    dt=.1
    b.defaultclock.dt=dt*b.ms
    neurons=pd.DataFrame({"root_id":["input","relay","output"],
        "cell_type":["LPLC2","relay","GF"],"role":["input","intermediate","output"],"version":"TEST"})
    edges=pd.DataFrame({"pre":["input","relay"],"post":["relay","output"],"synapses":[160,160],"signed_synapses":[160,160]})
    circuit=Circuit(neurons,edges,{"connectome_version":"TEST"})
    c=yaml.safe_load((Path(__file__).resolve().parents[1]/"configs/stimulation_v1.yaml").read_text())["simulation"]
    c.update(dt_ms=dt,baseline_ms=20,stimulus_ms=200,recovery_ms=40)
    times=np.arange(25.,220.,8.)
    events=np.column_stack([np.rint(times/dt).astype(int),np.zeros(len(times),dtype=int)])
    ours=simulate(circuit,["input"],events,c).spikes
    # Separate independent integrator and synaptic event scheduler.
    group=b.NeuronGroup(3,"""dv/dt=(-52*mV-v+g)/(20*ms) : volt (unless refractory)
        dg/dt=-g/(5*ms) : volt (unless refractory)
        rfc : second""",threshold="v>-45*mV",reset="v=-52*mV;g=0*mV",refractory="rfc",method="exact")
    group.v=-52*b.mV
    group.rfc=2.2*b.ms
    group[0].rfc=0*b.ms
    syn=b.Synapses(group,group,on_pre="g_post+=44*mV",delay=1.8*b.ms)
    syn.connect(i=[0,1],j=[1,2])
    source=b.SpikeGeneratorGroup(1,np.zeros(len(times),dtype=int),times*b.ms)
    drive=b.Synapses(source,group,on_pre="v_post+=68.75*mV")
    drive.connect(i=[0],j=[0])
    monitor=b.SpikeMonitor(group)
    b.Network(group,syn,source,drive,monitor).run(260*b.ms)
    theirs={name:np.asarray(monitor.spike_trains()[i]/b.ms) for i,name in enumerate(circuit.ids)}
    for name in ["relay","output"]:
        actual=ours[ours.root_id.eq(name)].time_ms.to_numpy()
        assert len(actual)>0
        assert len(actual)==len(theirs[name])
        # Different external-event scheduling moves timestamps by at most two dt.
        np.testing.assert_allclose(actual,theirs[name],atol=2*dt+1e-8,rtol=0)
