"""Final public checks, optionally including private-cache historical audits."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import subprocess,sys,time,shutil
from news_flywire.provenance import write_json,sha256


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,default=Path('results/final_release/validation'))
 p.add_argument('--full',action='store_true',help='Require original private publisher, feature and upstream caches')
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if (a.out/'summary.json').exists():raise ValueError('Validation already sealed; use a fresh output folder')
 py=sys.executable
 jobs=[('tests',[py,'-m','pytest','-q']),('article_845',[py,'scripts/audit_article_bridge.py']),('matched_120',[py,'scripts/audit_matched_controls.py'])]
 if a.full:
  snapshot=Path('data/corpus/20261109_v2');copy=Path('data/raw/finalization/corpus_audit')
  if not copy.exists():shutil.copytree(snapshot,copy)
  jobs += [('corpus_8128',[py,'scripts/audit_large_corpus.py','--out','data/raw/finalization/corpus_audit','--raw','data/raw/corpus/20261109_v2']),
   ('legacy_330',[py,'scripts/verify_outputs.py']),('v8',[py,'scripts/audit_new_article_diagnostics.py']),
   ('v9',[py,'scripts/audit_risk_calibration.py']),('v10',[py,'scripts/audit_context_risk.py'])]
 def run(job):
  name,command=job;start=time.monotonic();log=a.out/(name+'.log')
  historical=Path('results/verification.json')
  preserved=historical.read_bytes() if name=='legacy_330' and historical.exists() else None
  try:
   with log.open('w') as f:r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
   if name=='legacy_330' and historical.exists():shutil.copyfile(historical,a.out/'legacy_verification.json')
  finally:
   if preserved is not None:historical.write_bytes(preserved)
  result=dict(name=name,command=command,exit_code=r.returncode,passed=r.returncode==0,seconds=round(time.monotonic()-start,3),log=str(log),log_sha256=sha256(log))
  print(name,'PASS' if r.returncode==0 else 'FAIL',flush=True)
  return result
 results=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for future in as_completed([pool.submit(run,j) for j in jobs]):results.append(future.result())
 results.sort(key=lambda r:r['name']);passed=all(r['passed'] for r in results)
 write_json(a.out/'summary.json',dict(passed=passed,full_private_audits=a.full,checks=results,
  validator_sha256=sha256(__file__),scope='Code, source provenance and numerical reproducibility; not label truth or biological validity'))
 if not passed:raise SystemExit(1)


if __name__=='__main__':main()
