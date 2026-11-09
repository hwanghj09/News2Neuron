import argparse,json
from news_flywire.risk_v5 import prepare,train,packet,evaluate
p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','packet','evaluate']);a=p.parse_args()
if a.stage=='prepare':prepare()
elif a.stage=='train':train()
elif a.stage=='packet':print(json.dumps(packet(),ensure_ascii=False,indent=2))
else:evaluate()
