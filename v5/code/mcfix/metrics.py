"""Unknown-region semantic metrics and non-wrapping six-face boundaries."""
import numpy as np

def completion_metrics(gt_blocks,pred_blocks,gt_states,pred_states,known,valid=None):
    arrays=[np.asarray(v) for v in (gt_blocks,pred_blocks,gt_states,pred_states,known)]
    if len({a.shape for a in arrays})!=1:raise ValueError('Shape mismatch')
    g,p,gs,ps,known=arrays
    if known.dtype!=bool:raise ValueError('Mask must be boolean')
    unknown=~known;valid=np.ones_like(known) if valid is None else np.asarray(valid,dtype=bool)
    union=((g!=0)|(p!=0))&unknown;inter=(g!=0)&(p!=0)&unknown
    solid=(g!=0)&unknown;state_valid=solid&valid
    if valid.shape!=known.shape:raise ValueError('Validity mask shape mismatch')
    def ratio(a,b):return float(a)/int(b) if b else None
    expected=[];actual=[]
    for axis in range(g.ndim):
        for direction in [-1,1]:
            src=[slice(None)]*g.ndim;dst=src.copy()
            src[axis]=slice(1,None) if direction==1 else slice(None,-1)
            dst[axis]=slice(None,-1) if direction==1 else slice(1,None)
            src,dst=tuple(src),tuple(dst)
            edge=unknown[src]&known[dst]&(g[dst]!=0)
            expected.extend((g[src][edge]!=0).tolist());actual.extend((p[src][edge]!=0).tolist())
    exp=np.asarray(expected,bool);act=np.asarray(actual,bool)
    return {'unknown_occupancy_iou':ratio(inter.sum(),union.sum()),
            'unknown_class_accuracy_on_gt_solid':ratio(((g==p)&solid).sum(),solid.sum()),
            'unknown_full_state_accuracy':ratio(((gs==ps)&state_valid).sum(),state_valid.sum()),
            'boundary_adjacency_accuracy':ratio((exp==act).sum(),len(exp)),
            'boundary_edge_count':len(exp),
            'known_blocks_exact':bool(np.array_equal(g[known],p[known])),
            'known_states_exact':bool(np.array_equal(gs[known],ps[known])),
            'state_evaluated_cells':int(state_valid.sum())}
