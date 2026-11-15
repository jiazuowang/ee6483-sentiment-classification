"""Generate submission.csv using the trained local model, without retraining."""
import argparse
from pathlib import Path
import pandas as pd
import joblib
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'data/test.json');p.add_argument('--output',type=Path,default=ROOT/'submission.csv');a=p.parse_args()
model_path = ROOT/'results/sentiment_model.joblib'
if not model_path.is_file():
    raise SystemExit('No fitted model is distributed. Run code/run_experiments.py with authorized local data first.')
model=joblib.load(model_path)
test=pd.read_json(a.input);pred=model.predict(test.reviews)
pd.DataFrame({'reviews':test.reviews,'predictions':pred.astype(int)}).to_csv(a.output,index=False)
print(f'Saved {len(pred)} predictions to {a.output}')
