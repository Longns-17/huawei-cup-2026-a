"""Lazy proposal schedule specified by FINAL IDEA; budgets count admissions."""

from .construction import budget_coarsening, local_coarsening, seed
from .ordering import cache_stagger, release_priority
from .pricing import price_assignment

BUDGET = {1: 4, 2: 4, 3: 7}


def proposal_stream(problem, cores, scene, hardware, residency):
    if scene == 1:
        fine, diagnostic, _ = seed(problem, cores, 1, 2, False, hardware, residency)
        yield "fine", fine, diagnostic
        yield "local", *local_coarsening(problem, cores, hardware, residency)
        yield "budget_0.10", *budget_coarsening(problem, cores, 10, hardware, residency)
        yield "budget_0.05", *budget_coarsening(problem, cores, 5, hardware, residency)
        return
    anchor, diagnostic, _ = seed(problem, cores, scene, 2, True, hardware, residency)
    yield "a2_b1", anchor, diagnostic
    if scene == 3:
        yield "cache_price", *price_assignment(problem, anchor.units, cores, hardware, True, 1)
    yield "traffic_price", *price_assignment(problem, anchor.units, cores, hardware, False, 1)
    yield "pipeline_finish", *price_assignment(problem, anchor.units, cores, hardware, False, 0)
    alternate, diagnostic, _ = seed(problem, cores, scene, 4, False, hardware, residency)
    yield "a4_b0", alternate, diagnostic
    if scene == 3:
        yield "cache_order_1", *cache_stagger(problem, anchor, 2, hardware.cache_capacity)
    yield "release_order_8", *release_priority(problem, anchor, 8)
    if scene == 3:
        yield "cache_order_2", *cache_stagger(problem, anchor, 4, hardware.cache_capacity)
        yield "cache_order_3", *cache_stagger(problem, anchor, 8, hardware.cache_capacity)
    yield "release_order_4", *release_priority(problem, anchor, 4)
    for alpha, repair in ((2, False), (4, True)):
        candidate, diagnostic, _ = seed(problem, cores, scene, alpha, repair, hardware, residency)
        yield f"a{alpha}_b{int(repair)}", candidate, diagnostic


def admitted_proposals(problem, cores, scene, hardware, residency):
    seen = {}
    admitted = 0
    for label, placement, diagnostic in proposal_stream(problem, cores, scene, hardware, residency):
        signature = placement.structure()
        duplicate = seen[signature] if signature in seen else None
        yield dict(label=label, placement=placement, diagnostic=diagnostic, duplicate_of=duplicate)
        if duplicate is None:
            seen[signature] = label
            admitted += 1
            if admitted == BUDGET[scene]:
                break
