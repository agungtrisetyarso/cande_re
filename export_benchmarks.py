#!/usr/bin/env python3
"""
Writes the two benchmarks as self-contained data files, benchmarks/P.json and benchmarks/R.json.
Each file holds the compartmental matrix K, the feed vector f, the unmeasured net-transfer edges,
the organic stream behind each edge, the distribution coefficients, the organic volume fractions,
the units, and the provenance of every parameter. Nothing else is needed to rebuild the plant.
"""
import json, os
from benchmarks import build_path, build_ring, D_LYON, RING

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "..", "benchmarks")
os.makedirs(OUT, exist_ok=True)


def dump(name, pl, title, provenance, extra=None):
    d = dict(name=name, title=title, n_stages=pl.N, n_unmeasured_edges=pl.E, units=pl.units,
             model="dm/dt = -K m + f ; stages are numbered from 0 in this file and from 1 in the paper",
             K=pl.K.tolist(), f=pl.f.tolist(), edges_tail_head=[list(e) for e in pl.edges],
             organic_stream_source_sink=[list(e) for e in pl.org],
             distribution_coefficient=pl.Dc.tolist(), organic_volume_fraction=pl.phi_org.tolist(),
             provenance=provenance)
    if extra: d.update(extra)
    with open(os.path.join(OUT, name + ".json"), "w") as fh: json.dump(d, fh, indent=1)
    print("wrote benchmarks/%s.json  (%d stages, %d edges)" % (name, pl.N, pl.E))


dump("P", build_path(), "Synthetic 40-stage counter-current cascade",
     dict(status="SYNTHETIC. Every parameter is chosen by the authors; none is a plant or laboratory value.",
          distribution_coefficient="linear from 1.30 (stage 1) to 0.80 (stage 40), chosen",
          hold_ups="unit aqueous and organic hold-up in every stage, chosen",
          flows="unit aqueous and organic flow, chosen",
          feed="unit feed to stage 1, chosen"))

dump("R", build_ring(), "18-stage closed-loop Pr/Nd circuit (Nd balance)",
     dict(status="LABORATORY FLOWSHEET WITH ASSUMED HYDRAULICS. Not plant data.",
          source="K. L. Lyon, M.S. thesis, University of Idaho, 2016, Tables 3 and 17",
          from_source=dict(stage_counts="2 extraction, 12 scrubbing, 4 stripping",
                           vessel_volumes_mL="mixer 120, settler 480",
                           distribution_coefficient_Nd="measured values for stages 1-16"),
          assumed=dict(distribution_coefficient_stages_17_18="value of stage 16",
                       phase_split="aqueous and organic each 300 mL per stage",
                       flows_mL_min="organic, feed, scrub and strip each 5",
                       feed_Nd_g_L=RING["c_feed_g_L"], scrub_Nd_g_L=RING["c_scrub_g_L"],
                       organic_loop="closed: stripped organic of stage 18 returns to stage 1")),
     extra=dict(D_Nd_as_tabulated=D_LYON, hydraulics=RING))
