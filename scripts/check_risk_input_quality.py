"""Read-only full-corpus check; the preregistered v6 audit is never filtered."""
from news_flywire.risk_v4 import load_news,body_for
from news_flywire.risk_input_quality import incomplete_numeric_draft
from news_flywire.provenance import write_json

news=load_news();records=[]
for row in news.itertuples():
    if incomplete_numeric_draft(body_for(row)):
        records.append({'news_id':row.news_id,'source':row.source,'url':row.url,
            'assessment_body_sha256':row.assessment_body_sha256,'reason':'unfilled_numeric_broadcast_draft'})
write_json('data/risk/20261109_v6/input_quality_overlay.json',{'all_articles_checked':len(news),
    'flagged':len(records),'records':records,'v6_primary_cases_excluded':0,
    'model_inputs_or_predictions_changed':False,'no_missing_numeric_values_invented':True})
print('Incomplete numeric drafts',len(records))
