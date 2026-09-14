p = "/data/redstone_v4/work/_orchestrate_v4.sh"
src = open(p).read()
# make baseline eval non-fatal
old1 = """ATTN_BACKEND=sdpa HF_HUB_OFFLINE=1 $PY $WORK/_eval_v4_inpaint.py \\
  /data/redstone_v4/work/v4_init.pt $WORK/baseline_eval.json 24 \\
  >> $WORK/baseline_eval.log 2>&1
echo "[$(date +%T)] baseline eval done" | tee -a $WORK/orchestrator.log
tail -3 $WORK/baseline_eval.log >> $WORK/orchestrator.log 2>/dev/null || true"""
new1 = """ATTN_BACKEND=sdpa HF_HUB_OFFLINE=1 $PY $WORK/_eval_v4_inpaint.py \\
  /data/redstone_v4/work/v4_init.pt $WORK/baseline_eval.json 24 \\
  >> $WORK/baseline_eval.log 2>&1 || echo "[$(date +%T)] WARN baseline eval failed (non-fatal)" | tee -a $WORK/orchestrator.log
echo "[$(date +%T)] baseline eval done" | tee -a $WORK/orchestrator.log
tail -3 $WORK/baseline_eval.log >> $WORK/orchestrator.log 2>/dev/null || true"""
assert old1 in src
src = src.replace(old1, new1, 1)
# make final eval non-fatal
old2 = """ATTN_BACKEND=sdpa HF_HUB_OFFLINE=1 $PY $WORK/_eval_v4_inpaint.py \\
  "$CKPT" $WORK/final_eval.json 24 \\
  >> $WORK/final_eval.log 2>&1"""
new2 = """ATTN_BACKEND=sdpa HF_HUB_OFFLINE=1 $PY $WORK/_eval_v4_inpaint.py \\
  "$CKPT" $WORK/final_eval.json 24 \\
  >> $WORK/final_eval.log 2>&1 || echo "[$(date +%T)] WARN final eval failed" | tee -a $WORK/orchestrator.log"""
assert old2 in src
src = src.replace(old2, new2, 1)
open(p, "w").write(src)
print("orchestrator eval made non-fatal")
