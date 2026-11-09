import argparse,json
from news_flywire.risk_v4 import setup,features,packet,train,active_select,finalize,freeze
p=argparse.ArgumentParser()
p.add_argument('stage',choices=['setup','features','packet','train','select','freeze','finalize'])
p.add_argument('--role',choices=['development','fresh_random','fresh_stress'],default='development')
p.add_argument('--round',type=int,default=1)
a=p.parse_args()
if a.stage=='setup':setup()
elif a.stage=='features':features()
elif a.stage=='packet':print(json.dumps(packet(a.role,0 if a.role.startswith('fresh_') else a.round),ensure_ascii=False,indent=2))
elif a.stage=='train':train(a.round)
elif a.stage=='select':active_select()
elif a.stage=='freeze':freeze()
else:finalize()
